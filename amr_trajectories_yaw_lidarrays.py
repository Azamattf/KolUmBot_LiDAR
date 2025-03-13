
# # Animate AMR movement with LiDAR rays in the x-y plane from JSON step files

# In[1]:

import os
import json
import csv
import plotly.graph_objects as go
import pandas as pd
from natsort import natsorted
from collections import defaultdict
from tqdm import tqdm
import numpy as np
from scipy.spatial.transform import Rotation


# In[2]:


def main():
    # Data path

    path_to_dataset = "d:\\Works_Experience\\FML_Florian\\Kolumbot\\Lidar_visualization\\Unity_v3s\\"
    path_to_images = os.path.join(path_to_dataset, "sequence.0_json_only")
    save_path = os.path.join(path_to_dataset, "Export")
    path_to_sem_def = os.path.join(path_to_dataset, "class_definition_semantic_segmentation.json")
      
    # Process JSON files

    frame_step = 10

    json_files = load_json_files(path_to_images, frame_step)
    robot_data = read_data(json_files, save_path)

    class_def = load_class_definitions(path_to_sem_def)

    # Animate robot movement and LiDAR rays
    visualizer = Visualizer(robot_data, class_def)
    visualizer.animate()
    visualizer.save_animation(save_path)


# %% Class Definitions
def load_class_definitions(filepath):
    with open(filepath, 'r') as f:
        class_def = json.load(f)
    
    """
    label_map = {}
    color_map = {}
    
    
    for entry in class_def["m_LabelEntries"]:
        label = entry["label"]
        r = int(entry["color"]["r"] * 255)
        g = int(entry["color"]["g"] * 255)
        b = int(entry["color"]["b"] * 255)
        label_map[len(label_map)] = label
        color_map[label] = f'rgb({r},{g},{b})'
    """

    return class_def

# %% Visualization

class Visualizer:
    def __init__(self, robot_data, class_definitions):
        self.robot_data = robot_data
        self.class_definitions = class_definitions
        self.fig = go.Figure()
        
        self.robot_shapes = {}  # to store robot shape objects
        self.pointcloud_traces = []  # to store pointcloud traces
        self.time_stamps = sorted(list(robot_data.values())[0]["timestamps"])  # Get all timestamps

    def map_classes_to_colors(self, class_labels):
        """ Map pointcloud classes to colors using the semantic segmentation file. """
        class_colors = []
        for label in class_labels:
            color = self.class_definitions.get(label, {"color": {"r": 0, "g": 0, "b": 0, "a": 1.0}})["color"]
            rgb = f'rgba({int(color["r"] * 255)}, {int(color["g"] * 255)}, {int(color["b"] * 255)}, {color["a"]})'
            class_colors.append(rgb)
        return class_colors

    def update_traces(self, frame_idx):
        """ Update robot and pointcloud traces for a given frame index (timestamp). """
        # Update robot positions (trapezoids)
        for amr_id, data in self.robot_data.items():
            timestamp = data["timestamps"][frame_idx]
            robot_position = data["robot_positions"][frame_idx]
            robot_orientation = data["robot_orientations"][frame_idx]

            # Update the robot's position and orientation
            robot_trace = self.robot_shapes[amr_id]
            robot_trace.update(
                x=[robot_position[0]],
                y=[robot_position[1]],
                text=[f"{amr_id} @ {timestamp}"],
            )

        # Update point clouds (markers)
        for amr_id, data in self.robot_data.items():
            pointclouds = data["pointclouds"][frame_idx]
            pointcloud_x = [p["x"] for p in pointclouds]
            pointcloud_z = [p["z"] for p in pointclouds]
            pointcloud_classes = [p["class"] for p in pointclouds]
            colors = self.map_classes_to_colors(pointcloud_classes)

            # Add or update pointcloud trace
            pointcloud_trace = self.pointcloud_traces[amr_id]
            pointcloud_trace.update(
                x=pointcloud_x,
                y=pointcloud_z,
                marker=dict(color=colors)
            )

    def animate(self):
        """ Create animation using the robot data and pointclouds. """
        # Create initial robot shapes and pointcloud traces for the first frame
        for amr_id, data in self.robot_data.items():
            # Initial robot shapes (trapezoids)
            robot_position = data["robot_positions"][0]
            robot_orientation = data["robot_orientations"][0]
            self.robot_shapes[amr_id] = go.Scatter(
                x=[robot_position[0]],
                y=[robot_position[1]],
                mode="markers+text",
                text=[f"{amr_id}"],
                marker=dict(symbol="triangle-right", size=10, color="blue"),
                name=amr_id
            )
            self.fig.add_trace(self.robot_shapes[amr_id])

            # Initial pointcloud traces (markers)
            pointclouds = data["pointclouds"][0]
            pointcloud_x = [p["x"] for p in pointclouds]
            pointcloud_z = [p["z"] for p in pointclouds]
            pointcloud_classes = [p["class"] for p in pointclouds]
            colors = self.map_classes_to_colors(pointcloud_classes)

            pointcloud_trace = go.Scatter(
                x=pointcloud_x,
                y=pointcloud_z,
                mode="markers",
                marker=dict(color=colors, size=5, opacity=0.7),
                showlegend=False
            )
            self.pointcloud_traces.append(pointcloud_trace)
            self.fig.add_trace(pointcloud_trace)

        # Define frames for animation (one frame per timestamp)
        frames = []
        for idx in range(len(self.time_stamps)):
            frame = go.Frame(
                data=[self.robot_shapes[amr_id] for amr_id in self.robot_shapes] + self.pointcloud_traces,
                name=str(self.time_stamps[idx])
            )
            frames.append(frame)

        self.fig.frames = frames

        # Add slider for time control
        self.fig.update_layout(
            sliders=[{
                'steps': [{
                    'args': [[str(t)], {'frame': {'duration': 500, 'redraw': True}, 'mode': 'immediate'}],
                    'label': f'{t}',
                    'method': 'animate'
                } for t in self.time_stamps]
            }],
            updatemenus=[{
                'buttons': [{
                    'args': [None, {'frame': {'duration': 500, 'redraw': True}, 'fromcurrent': True}],
                    'label': 'Play',
                    'method': 'animate'
                }]
            }]
        )

        # Set axis limits based on maximum point cloud coordinates
        max_x = max([max([p["x"] for p in data["pointclouds"]]) for data in self.robot_data.values()])
        max_z = max([max([p["z"] for p in data["pointclouds"]]) for data in self.robot_data.values()])
        self.fig.update_layout(
            xaxis=dict(range=[0, max_x * 1.1]),
            yaxis=dict(range=[0, max_z * 1.1])
        )

    def save_animation(self, savepath):
        """ Save the animation to an HTML file. """
        os.makedirs(savepath, exist_ok=True)
        filepath = os.path.join(savepath, "AMR_Animation_with_Lidarrays.html")
        self.fig.write_html(filepath)
        print(f"\n✅ Animation saved to {filepath}")



# %% Convert from local Unity point cloud to global Unity Cartersian

""" 
Transform lidar points in local CoSy to global Unity Cartesian CoSy
LiDAR positions and rotations are global
Cloud point angles are in degrees
"""
def transform_lidar_points(
    ranges: np.ndarray,
    angles: np.ndarray,
    lidar_position: np.ndarray,
    lidar_quaternion: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:

    # 1. Convert polar to local Cartesian coordinates
    theta_rad = np.deg2rad(angles)
    x_local = ranges * np.cos(theta_rad)
    z_local = ranges * np.sin(theta_rad)
    
    # Stack coordinates and add y=0 for 3D rotation
    points_local = np.vstack((x_local, np.zeros_like(x_local), z_local))
    
    # 2. Create rotation from quaternion
    R = Rotation.from_quat(lidar_quaternion)
    
    # 3. Apply rotation
    points_rotated = R.apply(points_local.T)
    
    # 4. Apply translation
    x_global = points_rotated[:, 0] + lidar_position[0]
    # y_global = points_rotated[:, 1] + lidar_position[1]
    z_global = points_rotated[:, 2] + lidar_position[2]
    
    return x_global, z_global   

# %% Load JSONs


def load_json_files(path_to_jsons, frame_step):
    data = []
    json_files = [f for f in natsorted(os.listdir(path_to_jsons)) if f.endswith(".json")]
    sampled_files = json_files[::frame_step]  # Sample files based on frame_step

    # Check first 5 file names
    print("\nFirst 5 files:")
    print(f"{json_files[:5]} ...")  # Print first 5 positions for each robot

    print("\nFirst 5 sampled files:")
    print(f"{sampled_files[:5]} ...")  # Print first 5 positions for each robot

    print("")

    for i, filename in tqdm(list(enumerate(json_files)), ascii="░▒█", desc="Loading JSON files", unit=" file"):
        if i % frame_step != 0:  # Skip frames based on step
            continue

        file_path = os.path.join(path_to_jsons, filename)

        with open(file_path, "r", encoding='utf-8') as f:
            try:
                data.append(json.load(f))
            except json.JSONDecodeError:
                print(f"Error decoding {filename}, skipping.")
                continue

    print(f"\nLoaded {len(data)} frames (Step = {frame_step})")
                                
    return data


# %% Extract positions, yaw and point cloud

def read_data(data, savepath, filename="first10instances.json"):
    robot_data = defaultdict(lambda: {
        "timestamps": [],
        "robot_positions": [],
        "robot_orientations": [],
        "pointclouds": []
    })

    for frame_index, frame in tqdm(enumerate(data), ascii="░▒█", desc="Processing JSON files", unit="file"):
        timestamp = frame.get("timestamp", 0)

        # Filter out 2D lidars
        lidars = [o for o in frame.get("captures", []) if o.get('@type') == 'type.custom/solo.2DLidar']

        for lidar in lidars:
            amr_id = "_".join(lidar["id"].split("_")[:2])  # e.g., "AMR_1"

            # Initialize per-lidar/AMR fields
            robot_position = [None, None]
            robot_orientation = [None, None]

            # Extract front lidar pose (robot pose)
            if 'front' in lidar['id']:
                robot_position_raw = lidar.get("robotPosition", [])
                robot_yaw_raw = lidar.get("robotRotationEuler", [])  # in radians

                if len(robot_position_raw) == 3:
                    # Take X and Z only (floor plan view)
                    robot_position = [robot_position_raw[0], robot_position_raw[2]]

                robot_orientation = [np.cos(robot_yaw_raw), np.sin(robot_yaw_raw)]

                # Append data to the structure
                robot_data[amr_id]["timestamps"].append(timestamp)
                robot_data[amr_id]["robot_positions"].append(robot_position)
                robot_data[amr_id]["robot_orientations"].append(robot_orientation)
            


            # Global lidar pose (for transforming points)
            lidar_position = np.array(lidar.get("globalPosition", [0, 0, 0]))   # global position
            lidar_quaternion = np.array(lidar.get("globalRotation", [0, 0, 0, 1]))  # global rotation

            # Process annotations (actual lidar scan data)
            for annot in lidar.get("annotations", []):
                ranges = np.array(annot.get("ranges", []))
                angles = np.array(annot.get("angles", []))
                classes = np.array(annot.get("object_classes", []))

                # Transform points to global frame
                pointcloud_x, pointcloud_z = transform_lidar_points(
                    ranges, angles, lidar_position, lidar_quaternion
                )

                pointcloud = [{"x": x, "z": z, "class": c} for x, z, c in zip(pointcloud_x, pointcloud_z, classes)]
            robot_data[amr_id]["pointclouds"].append(pointcloud)
    
    # Save first 10 instances as JSON
    first_10_instances = dict(list(robot_data.items())[:10])

    os.makedirs(savepath, exist_ok=True)
    filepath = os.path.join(savepath, filename)

    with open(filepath, 'w') as f:
        json.dump(first_10_instances, f, indent=4)
    print(f"\n✅ Robot yaw data saved to {filepath}")

    return robot_data




# In[6]:


if __name__ == "__main__":
    main()
