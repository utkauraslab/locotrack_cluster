import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "locotrack_pytorch"))
import uuid
import spaces
import math
import csv
import gradio as gr
import mediapy
import numpy as np
import cv2
import matplotlib
import torch
from sklearn.decomposition import PCA

from locotrack_pytorch.models.locotrack_model import load_model, FeatureGrids
from viz_utils import paint_point_track

# parameter config
CLUSTER_COLORS = {
    "A": (255, 0, 0),    # red
    "B": (0, 255, 0),    # green
    "C": (0, 0, 255)     # blue
}
CLUSTERS = ["A", "B", "C"]
PREVIEW_WIDTH = 768
VIDEO_INPUT_RESO = (256, 256)
POINT_SIZE = 4
FRAME_LIMIT = 300
WEIGHTS_PATH = {
    "small": "./weights/locotrack_small.ckpt",
    "base": "./weights/locotrack_base.ckpt",
}

def draw_rigid_transform(frame, translation, rotation, color, length=30):
    """draw rigid body transformation"""
    center = tuple(map(int, translation))
    dx = length * math.cos(rotation)
    dy = length * math.sin(rotation)
    
    end_point = (int(center[0] + dx), int(center[1] + dy))
    frame = cv2.arrowedLine(frame, center, end_point, color, 2, tipLength=0.3)
    
    ortho_rot = rotation + math.pi/2
    dx_ortho = length/2 * math.cos(ortho_rot)
    dy_ortho = length/2 * math.sin(ortho_rot)
    end_ortho = (int(center[0] + dx_ortho), int(center[1] + dy_ortho))
    frame = cv2.line(frame, center, end_ortho, color, 1)
    
    return frame

def get_point(frame_num, video_queried_preview, query_points, query_points_color, query_count, current_cluster, evt: gr.SelectData):
    current_frame = video_queried_preview[int(frame_num)]
    
    cluster_points = query_points[current_cluster]
    cluster_colors = query_points_color[current_cluster]
    
    cluster_points[int(frame_num)].append((evt.index[0], evt.index[1], frame_num))
    cluster_colors[int(frame_num)].append(CLUSTER_COLORS[current_cluster])

    x, y = evt.index
    current_frame_draw = cv2.circle(current_frame, (x, y), POINT_SIZE, CLUSTER_COLORS[current_cluster], -1)
    video_queried_preview[int(frame_num)] = current_frame_draw
    
    query_count += 1
    return (
        current_frame_draw,
        video_queried_preview,
        query_points,
        query_points_color,
        query_count
    )

def undo_point(frame_num, video_preview, video_queried_preview, query_points, query_points_color, query_count, current_cluster):
    cluster_points = query_points[current_cluster]
    cluster_colors = query_points_color[current_cluster]
    
    if len(cluster_points[int(frame_num)]) == 0:
        return (
            video_queried_preview[int(frame_num)],
            video_queried_preview,
            query_points,
            query_points_color,
            query_count
        )

    cluster_points[int(frame_num)].pop(-1)
    cluster_colors[int(frame_num)].pop(-1)

    current_frame_draw = video_preview[int(frame_num)].copy()
    for cluster in CLUSTERS:
        for point, color in zip(query_points[cluster][int(frame_num)], query_points_color[cluster][int(frame_num)]):
            x, y, _ = point
            current_frame_draw = cv2.circle(current_frame_draw, (x, y), POINT_SIZE, color, -1)

    video_queried_preview[int(frame_num)] = current_frame_draw
    query_count -= 1
    return (
        current_frame_draw,
        video_queried_preview,
        query_points,
        query_points_color,
        query_count
    )

def clear_frame_fn(frame_num, video_preview, video_queried_preview, query_points, query_points_color, query_count, current_cluster):
    cluster_points = query_points[current_cluster]
    cluster_colors = query_points_color[current_cluster]
    
    query_count -= len(cluster_points[int(frame_num)])
    cluster_points[int(frame_num)] = []
    cluster_colors[int(frame_num)] = []

    current_frame_draw = video_preview[int(frame_num)].copy()
    for cluster in CLUSTERS:
        for point, color in zip(query_points[cluster][int(frame_num)], query_points_color[cluster][int(frame_num)]):
            x, y, _ = point
            current_frame_draw = cv2.circle(current_frame_draw, (x, y), POINT_SIZE, color, -1)

    video_queried_preview[int(frame_num)] = current_frame_draw
    return (
        current_frame_draw,
        video_queried_preview,
        query_points,
        query_points_color,
        query_count
    )

def clear_all_fn(frame_num, video_preview):
    return (
        video_preview[int(frame_num)],
        video_preview.copy(),
        {c: [[] for _ in range(len(video_preview))] for c in CLUSTERS},
        {c: [[] for _ in range(len(video_preview))] for c in CLUSTERS},
        0
    )

def choose_frame(frame_num, video_preview_array):
    return video_preview_array[int(frame_num)]

@spaces.GPU
def extract_feature(video_input, model_size="small"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float16

    model = load_model(WEIGHTS_PATH[model_size], model_size=model_size).to(device)

    video_input = (video_input / 255.0) * 2 - 1
    video_input = torch.tensor(video_input).unsqueeze(0).to(device, dtype)

    with torch.autocast(device_type=device, dtype=dtype):
        with torch.no_grad():
            feature = model.get_feature_grids(video_input)

    feature = FeatureGrids(
        lowres=(feature.lowres[-1].cpu(),),
        hires=(feature.hires[-1].cpu(),),
        highest=(feature.highest[-1].cpu(),),
        resolutions=(feature.resolutions[-1],),
    )
    return feature

def preprocess_video_input(video_path, model_size):
    video_arr = mediapy.read_video(video_path)
    video_fps = video_arr.metadata.fps
    num_frames = video_arr.shape[0]
    if num_frames > FRAME_LIMIT:
        gr.Warning(f"The video is too long. Only the first {FRAME_LIMIT} frames will be used.", duration=5)
        video_arr = video_arr[:FRAME_LIMIT]
        num_frames = FRAME_LIMIT

    height, width = video_arr.shape[1:3]
    new_height, new_width = int(PREVIEW_WIDTH * height / width), PREVIEW_WIDTH

    preview_video = mediapy.resize_video(video_arr, (new_height, new_width))
    input_video = mediapy.resize_video(video_arr, VIDEO_INPUT_RESO)

    preview_video = np.array(preview_video)
    input_video = np.array(input_video)

    video_feature = extract_feature(input_video, model_size)
    
    return (
        video_arr,
        preview_video,
        preview_video.copy(),
        input_video,
        video_feature,
        video_fps,
        gr.update(open=False),
        model_size,
        preview_video[0],
        gr.update(minimum=0, maximum=num_frames - 1, value=0, interactive=True),
        {c: [[] for _ in range(num_frames)] for c in CLUSTERS},
        {c: [[] for _ in range(num_frames)] for c in CLUSTERS},
        0,
        gr.update(interactive=True),
        gr.update(interactive=True),
        gr.update(interactive=True),
        gr.update(interactive=True),
    )

def compute_rigid_transforms(tracks, clusters_points, num_frames):
    transforms = {cluster: [None]*num_frames for cluster in CLUSTERS}
    
    for cluster in CLUSTERS:
        track_idx = 0
        for frame_idx in range(num_frames):
            frame_points = clusters_points[cluster][frame_idx]
            if len(frame_points) < 2:
                continue
                
            points = tracks[track_idx:track_idx+len(frame_points), frame_idx, :]
            points = np.array([(p[0], p[1]) for p in points])
            
            centroid = np.mean(points, axis=0)
            
            pca = PCA(n_components=2)
            pca.fit(points)
            components = pca.components_
            
            angle = math.atan2(components[0,1], components[0,0])
            
            transforms[cluster][frame_idx] = {
                "translation": centroid,
                "rotation": angle,
                "components": components
            }
            track_idx += len(frame_points)
                
    return transforms

@spaces.GPU
def track(
    model_size, 
    video_preview,
    video_input, 
    video_feature, 
    video_fps, 
    query_points,
    query_points_color, 
    query_count, 
):
    if query_count == 0:
        gr.Warning("Please add query points before tracking.", duration=5)
        return None, None
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float16

    video_feature = FeatureGrids(
        lowres=(video_feature.lowres[-1].to(device, dtype),),
        hires=(video_feature.hires[-1].to(device, dtype),),
        highest=(video_feature.highest[-1].to(device, dtype),),
        resolutions=(video_feature.resolutions[-1],),
    )

    all_points = []
    for cluster in CLUSTERS:
        for frame_points in query_points[cluster]:
            all_points.extend(frame_points)
    
    query_points_tensor = torch.tensor(all_points).float()
    query_points_tensor *= torch.tensor([
        VIDEO_INPUT_RESO[1], VIDEO_INPUT_RESO[0], 1
    ]) / torch.tensor([
        [video_preview.shape[2], video_preview.shape[1], 1]
    ])
    query_points_tensor = query_points_tensor[None].flip(-1).to(device, dtype)

    video_input = (video_input / 255.0) * 2 - 1
    video_input = torch.tensor(video_input).unsqueeze(0).to(device, dtype)

    model = load_model(WEIGHTS_PATH[model_size], model_size=model_size).to(device)
    with torch.autocast(device_type=device, dtype=dtype):
        with torch.no_grad():
            output = model(video_input, query_points_tensor, feature_grids=video_feature)

    tracks = output['tracks'][0].cpu()
    tracks = tracks * torch.tensor([
        video_preview.shape[2], video_preview.shape[1]
    ]) / torch.tensor([
        VIDEO_INPUT_RESO[1], VIDEO_INPUT_RESO[0]
    ])
    tracks = np.array(tracks)

    num_frames = video_preview.shape[0]
    transforms = compute_rigid_transforms(tracks, query_points, num_frames)

    painted_video = []
    for frame_idx in range(num_frames):
        frame = video_preview[frame_idx].copy()
        
        # draw all points
        for cluster in CLUSTERS:
            for point, color in zip(query_points[cluster][frame_idx], query_points_color[cluster][frame_idx]):
                x, y, _ = point
                frame = cv2.circle(frame, (x, y), POINT_SIZE, color, -1)
        
        # draw pose
        for cluster in CLUSTERS:
            transform = transforms[cluster][frame_idx]
            if transform is not None:
                frame = draw_rigid_transform(
                    frame,
                    transform["translation"],
                    transform["rotation"],
                    CLUSTER_COLORS[cluster]
                )
        painted_video.append(frame)

    video_file_name = uuid.uuid4().hex + ".mp4"
    video_path = os.path.join(os.path.dirname(__file__), "tmp")
    video_file_path = os.path.join(video_path, video_file_name)
    os.makedirs(video_path, exist_ok=True)
    mediapy.write_video(video_file_path, painted_video, fps=video_fps)

    # generate CSV file
    csv_file_name = uuid.uuid4().hex + ".csv"
    csv_path = os.path.join(video_path, csv_file_name)
    
    # collect traj info
    all_points_info = []
    for cluster in CLUSTERS:
        for frame_idx in range(num_frames):
            points_in_frame = query_points[cluster][frame_idx]
            for point_idx, _ in enumerate(points_in_frame):
                all_points_info.append({
                    "cluster": cluster,
                    "original_frame": frame_idx,
                    "original_point_id": point_idx
                })
                
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['frame', 'cluster', 'point_id', 'x', 'y', 
                     'translation_x', 'translation_y', 'rotation_rad']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for track_idx in range(tracks.shape[0]):
            info = all_points_info[track_idx]
            cluster = info["cluster"]
            original_point_id = info["original_point_id"]
            
            for frame_idx in range(num_frames):
                x = tracks[track_idx, frame_idx, 0].item()
                y = tracks[track_idx, frame_idx, 1].item()
                
                transform = transforms[cluster][frame_idx] if frame_idx < len(transforms[cluster]) else None
                
                row_data = {
                    'frame': frame_idx,
                    'cluster': cluster,
                    'point_id': original_point_id,
                    'x': round(x, 2),
                    'y': round(y, 2),
                }
                
                if transform:
                    row_data.update({
                        'translation_x': round(transform["translation"][0], 2),
                        'translation_y': round(transform["translation"][1], 2),
                        'rotation_rad': round(transform["rotation"], 4)
                    })
                else:
                    row_data.update({
                        'translation_x': None,
                        'translation_y': None,
                        'rotation_rad': None
                    })
                
                writer.writerow(row_data)

    return video_file_path, csv_path

with gr.Blocks() as demo:
    video = gr.State()
    video_queried_preview = gr.State()
    video_preview = gr.State()
    video_input = gr.State()
    video_feature = gr.State()
    video_fps = gr.State(24)
    model_size = gr.State("small")
    query_points = gr.State()
    query_points_color = gr.State()
    query_count = gr.State(0)
    current_cluster = gr.State("A")

    gr.Markdown("# LocoTrack Cluster Demo")
    gr.Markdown("Track multiple object clusters with rigid transformation analysis")

    with gr.Accordion("Video Input", open=True) as video_in_drawer:
        model_size_selection = gr.Radio(
            label="Model Size",
            choices=["small", "base"],
            value="small",
        )
        video_in = gr.Video(label="Input Video", format="mp4")
        with gr.Row():
            example = gr.Examples(
                examples=[
                    ["./examples/bmx-bumps.mp4"],
                    ["./examples/bmx-trees.mp4"],
                    ["./examples/breakdance-flare.mp4"],
                    ["./examples/breakdance.mp4"],
                ],
                inputs=[video_in],
                examples_per_page=2
            )
            submit = gr.Button("Submit", interactive=False)

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Annotation Panel")
            with gr.Row():
                cluster_radio = gr.Radio(
                    choices=CLUSTERS,
                    value="A",
                    label="Select Cluster",
                    interactive=True
                )
            with gr.Row():
                query_frames = gr.Slider(
                    minimum=0, maximum=100, value=0, step=1, 
                    label="Frame Selector", interactive=False)
            with gr.Row():
                undo = gr.Button("Undo", interactive=False)
                clear_frame = gr.Button("Clear Frame", interactive=False)
                clear_all = gr.Button("Clear All", interactive=False)
            current_frame = gr.Image(
                label="Annotation Canvas", 
                type="numpy",
                interactive=False
            )
            
            track_button = gr.Button("Track", interactive=False)

        with gr.Column():
            gr.Markdown("### Tracking Results")
            csv_file = gr.File(label="Download Pose CSV", interactive=False)
            output_video = gr.Video(
                label="Output Visualization",
                interactive=False,
                autoplay=True,
                loop=True,
            )

    video_in.change(
        lambda x: gr.update(interactive=True) if x else gr.update(interactive=False),
        [video_in], [submit]
    )
    
    submit.click(
        preprocess_video_input,
        [video_in, model_size_selection],
        [
            video, video_preview, video_queried_preview, video_input, video_feature,
            video_fps, video_in_drawer, model_size, current_frame, query_frames,
            query_points, query_points_color, query_count, undo, clear_frame,
            clear_all, track_button
        ],
        queue=True
    )

    cluster_radio.change(
        lambda x: x,
        [cluster_radio], [current_cluster],
        queue=False
    )

    query_frames.change(
        choose_frame,
        [query_frames, video_queried_preview],
        [current_frame],
        queue=False
    )

    current_frame.select(
        get_point,
        [query_frames, video_queried_preview, query_points, query_points_color, query_count, current_cluster],
        [current_frame, video_queried_preview, query_points, query_points_color, query_count],
        queue=False
    )

    undo.click(
        undo_point,
        [query_frames, video_preview, video_queried_preview, query_points, query_points_color, query_count, current_cluster],
        [current_frame, video_queried_preview, query_points, query_points_color, query_count],
        queue=False
    )

    clear_frame.click(
        clear_frame_fn,
        [query_frames, video_preview, video_queried_preview, query_points, query_points_color, query_count, current_cluster],
        [current_frame, video_queried_preview, query_points, query_points_color, query_count],
        queue=False
    )

    clear_all.click(
        clear_all_fn,
        [query_frames, video_preview],
        [current_frame, video_queried_preview, query_points, query_points_color, query_count],
        queue=False
    )

    track_button.click(
        track,
        [model_size, video_preview, video_input, video_feature, video_fps, query_points, query_points_color, query_count],
        [output_video, csv_file],
        queue=True
    )

if __name__ == "__main__":
    demo.launch(show_api=False, show_error=True)