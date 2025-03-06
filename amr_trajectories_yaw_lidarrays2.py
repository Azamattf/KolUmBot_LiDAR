import os
import json
import plotly.graph_objects as go
import pandas as pd
from natsort import natsorted
from collections import defaultdict
from tqdm import tqdm
import numpy as np
from scipy.spatial.transform import Rotation


def main():
    # Data path
    path_to_dataset = "d:\\Works_Experience\\FML_Florian\\Kolumbot\\Lidar_visualization\\Unity_v3s\\"
    path_to_images = os.path.join(path_to_dataset, "sequence.0_json_only")
    save_path = os.path.join(path_to_dataset, "Export")
    path_to_sem_def = os.path.join(path_to_dataset, "class_definition_semantic_segmentation.json")
    
    # Process JSON files
    frame_step = 50
    json_files = load_json_files(path_to_images, frame_step)
    
    # Load class definitions for point cloud coloring
    class_colors = load_class_colors(path_to_sem_def)
    
    # Extract data from JSON files
    position_data = read_data(json_files)
    
    # Animate robot movement and point clouds
    animation = AMRAnimation(position_data, class_colors)
    animation.create_animation(save_path)


class AMR:
    def __init__(self, name, data, color):
        self.name = name
        self.color = color
        self.timestamps = data["timestamps"]
        self.positions = data["robot_positions"]
        self.orientations = data["robot_orientations"]
        self.pointclouds = data["pointclouds"]
        self.x_values = []
        self.y_values = []
    
    def get_trace(self):
        # Create an initial trace for the AMR's path
        return go.Scatter(x=[], y=[], mode="lines+markers", name=self.name,
                         line=dict(color=self.color), marker=dict(color=self.color))
    
    def get_position_at_frame(self, frame_idx):
        # Get position and orientation at a specific frame
        if frame_idx < len(self.positions):
            x, y = self.positions[frame_idx]
            yaw_x, yaw_y = self.orientations[frame_idx]
            timestamp = self.timestamps[frame_idx]
            return x, y, yaw_x, yaw_y, timestamp
        elif len(self.positions) > 0:
            # Return last known position if beyond trajectory
            x, y = self.positions[-1]
            yaw_x, yaw_y = self.orientations[-1]
            timestamp = self.timestamps[-1]
            return x, y, yaw_x, yaw_y, timestamp
        else:
            # Fallback if trajectory is empty
            return 0, 0, 0, 0, 0
    
    def get_pointcloud_at_frame(self, frame_idx):
        # Get the point cloud data at a specific frame
        if frame_idx < len(self.pointclouds):
            return self.pointclouds[frame_idx]
        return []
    
    def create_robot_shape(self, x, y, yaw_x, yaw_y, size=1.0):
        # Create a rotated rectangle to represent the robot
        angle = np.arctan2(yaw_y, yaw_x)
        
        # Rectangle dimensions
        length = size * 2.0  # Robot length
        width = size * 1.5   # Robot width
        
        # Calculate corners of the rectangle
        corners = np.array([
            [-length/2, -width/2],  # back left
            [length/2, -width/2],   # front left
            [length/2, width/2],    # front right
            [-length/2, width/2],   # back right
            [-length/2, -width/2]   # back left (close the shape)
        ])
        
        # Rotate the corners
        rot_matrix = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ])
        rotated_corners = np.dot(corners, rot_matrix.T)
        
        # Translate to position
        rotated_corners[:, 0] += x
        rotated_corners[:, 1] += y
        
        return rotated_corners
        
    def get_frame(self, frame_idx, class_colors):
        # Generate a frame showing the AMR's position and point cloud
        traces = []
        
        # Update position history up to this frame
        self.x_values = []
        self.y_values = []
        
        for i in range(min(frame_idx + 1, len(self.positions))):
            x, y = self.positions[i]
            self.x_values.append(x)
            self.y_values.append(y)
            
        # Create a trace for the path
        path_trace = go.Scatter(
            x=self.x_values, y=self.y_values, mode="lines",
            line=dict(color=self.color, width=2), 
            name=f"{self.name} Path",
            hoverinfo="none"
        )
        traces.append(path_trace)
        
        # Get current position and orientation
        current_x, current_y, yaw_x, yaw_y, _ = self.get_position_at_frame(frame_idx)
        
        # Create a trace for the robot's current position as a rotated shape
        if current_x is not None and current_y is not None and yaw_x is not None and yaw_y is not None:
            robot_shape = self.create_robot_shape(current_x, current_y, yaw_x, yaw_y)
            
            robot_trace = go.Scatter(
                x=robot_shape[:, 0], y=robot_shape[:, 1], 
                mode="lines+fill", fill="toself",
                fillcolor=self.color, line=dict(color="black", width=1),
                name=f"{self.name}"
            )
            traces.append(robot_trace)
        
        # Add point cloud data
        pointcloud = self.get_pointcloud_at_frame(frame_idx)
        
        # Group point cloud by class for efficient rendering
        points_by_class = defaultdict(lambda: {"x": [], "y": []})
        
        for point in pointcloud:
            class_id = point["class"]
            points_by_class[class_id]["x"].append(point["x"])
            points_by_class[class_id]["y"].append(point["y"])
        
        # Create a trace for each point cloud class
        for class_id, points in points_by_class.items():
            # Get color for this class or use a default if not found
            point_color = class_colors.get(class_id, "gray")
            
            point_trace = go.Scatter(
                x=points["x"], y=points["y"], 
                mode="markers",
                marker=dict(color=point_color, size=3, opacity=0.7),
                name=f"Class {class_id}",
                hoverinfo="none"
            )
            traces.append(point_trace)
            
        return traces


class AMRAnimation:
    def __init__(self, robot_data, class_colors, fps=5, frame_step=1):
        self.robot_data = robot_data
        self.class_colors = class_colors
        self.fps = fps
        self.frame_step = frame_step
        self.amrs = self.create_amrs()
        self.x_range, self.y_range = self.compute_limits()
        self.max_frames = max(len(data["timestamps"]) for data in self.robot_data.values())
        
    def create_amrs(self):
        color_palette = ["blue", "red", "green", "purple", "orange", "brown", "cyan", "magenta"]
        return [AMR(name, data, color_palette[i % len(color_palette)]) 
                for i, (name, data) in enumerate(self.robot_data.items())]
    
    def compute_limits(self):
        # Find x and y limits across all robot positions
        x_positions = []
        y_positions = []
        
        for data in self.robot_data.values():
            for pos in data["robot_positions"]:
                if pos[0] is not None and pos[1] is not None:
                    x_positions.append(pos[0])
                    y_positions.append(pos[1])
        
        # Add buffer space
        buffer = 10
        x_min, x_max = min(x_positions) - buffer, max(x_positions) + buffer
        y_min, y_max = min(y_positions) - buffer, max(y_positions) + buffer
        
        return (x_min, x_max), (y_min, y_max)
    
    def get_timestamp_at_frame(self, frame_idx):
        if self.amrs:
            _, _, _, _, timestamp = self.amrs[0].get_position_at_frame(frame_idx)
            return timestamp
        return 0
    
    def format_timestamp(self, timestamp):
        minutes = int(timestamp // 60)
        seconds = int(timestamp % 60)
        return f"{minutes:02d}:{seconds:02d}"
    
    def create_animation(self, save_path):
        # Get initial traces for all AMRs
        initial_traces = []
        for amr in self.amrs:
            initial_traces.extend(amr.get_frame(0, self.class_colors))
        
        frames = []
        slider_steps = []
        
        for frame_idx in range(0, self.max_frames, self.frame_step):
            # Get updated traces for all AMRs including point clouds
            frame_data = []
            for amr in self.amrs:
                frame_data.extend(amr.get_frame(frame_idx, self.class_colors))
            
            # Get timestamp for this frame
            timestamp = self.get_timestamp_at_frame(frame_idx)
            formatted_time = self.format_timestamp(timestamp)
            frame_name = f"Frame_{frame_idx}"
            
            # Create time annotation
            time_annotation = {
                "text": f"Time: {timestamp:.2f} s",
                "xref": "paper", "yref": "paper",
                "x": 0.95, "y": 0.95,
                "showarrow": False,
                "font": {"size": 14, "color": "black"},
                "bgcolor": "white",
                "bordercolor": "black",
                "borderwidth": 1
            }
            
            # Create the frame
            frames.append(
                go.Frame(
                    data=frame_data,
                    name=frame_name,
                    layout=dict(annotations=[time_annotation])
                ))
            
            # Create slider step
            slider_steps.append({
                "args": [[frame_name], {"frame": {"duration": 1000 // self.fps, "redraw": True}, "mode": "immediate"}],
                "label": formatted_time,
                "method": "animate"
            })
        
        # Create the figure
        fig = go.Figure(
            data=initial_traces,
            layout=go.Layout(
                title="AMR Animation with Point Cloud",
                xaxis=dict(
                    title="X Position",
                    range=self.x_range,
                    zeroline=True,
                    scaleanchor="y",
                    scaleratio=1,
                    fixedrange=True
                ),
                yaxis=dict(
                    title="Y Position",
                    range=self.y_range,
                    zeroline=True,
                    scaleanchor="x",
                    scaleratio=1,
                    fixedrange=True
                ),
                updatemenus=[
                    {
                        "type": "buttons",
                        "buttons": [
                            {
                                "args": [None, {
                                    "frame": {"duration": 1000 // self.fps, "redraw": True},
                                    "fromcurrent": True,
                                    "transition": {"duration": 500, "easing": "cubic-in-out"}
                                }],
                                "label": "Play",
                                "method": "animate"
                            },
                            {
                                "args": [[None], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
                                "label": "Pause",
                                "method": "animate"
                            }
                        ],
                        "direction": "left",
                        "showactive": True,
                        "x": 0.1,
                        "xanchor": "right",
                        "y": 0.05,
                        "yanchor": "bottom",
                        "pad": {"t": 10, "r": 10},
                        "bgcolor": "lightgray",
                        "bordercolor": "gray",
                        "borderwidth": 1
                    }
                ],
                sliders=[{
                    "steps": slider_steps,
                    "currentvalue": {"prefix": "Time: ", "visible": True, "xanchor": "right"}
                }],
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            ),
            frames=frames
        )
        
        fig.show()
        
        # Save animation as HTML
        os.makedirs(save_path, exist_ok=True)
        filepath = os.path.join(save_path, "AMR_Animation_with_PointCloud.html")
        fig.write_html(filepath)


def load_class_colors(path_to_sem_def):
    """Load class definitions and assign colors to each class ID"""
    try:
        with open(path_to_sem_def, 'r') as f:
            class_def = json.load(f)
            
        # Create a color mapping for each class
        color_map = {}
        
        for i, entry in enumerate(class_def.get("m_LabelEntries", [])):
            # Extract RGBA color from the definition
            rgba = entry.get("color", {"r": 0, "g": 0, "b": 0, "a": 1})
            
            # Convert to RGB hex string for Plotly
            r = int(rgba["r"] * 255)
            g = int(rgba["g"] * 255)
            b = int(rgba["b"] * 255)
            
            color_hex = f"rgb({r},{g},{b})"
            
            # Map class ID to color and name
            color_map[i] = color_hex
            
        return color_map
    except Exception as e:
        print(f"Error loading class definitions: {e}")
        # Return default colors if file can't be loaded
        return {i: f"hsl({(i * 30) % 360}, 70%, 50%)" for i in range(20)}


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
    
    # 5. Convert to ROS coordinate system
    x_ros = x_global
    y_ros = z_global
    # z_ros = y_global  # Unity Y -> ROS Z
    
    return x_ros, y_ros


def read_data(data):
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
            robot_position = [0, 0]  # Default position
            robot_orientation = [1, 0]  # Default orientation (facing right)

            # Extract front lidar pose (robot pose)
            if 'front' in lidar['id']:
                robot_position_raw = lidar.get("robotPosition", [])
                robot_yaw_raw = lidar.get("robotRotationEuler", [])

                if len(robot_position_raw) == 3:
                    # Take X and Z only (floor plan view)
                    robot_position = [robot_position_raw[0], robot_position_raw[2]]

                # FIX: Changed to handle robot_yaw_raw correctly if it's a float or a list of floats
                if isinstance(robot_yaw_raw, list) and len(robot_yaw_raw) > 0:
                    # It's a list with at least one element
                    yaw_rad = np.radians(robot_yaw_raw[0])  # Convert to radians
                    robot_orientation = [np.cos(yaw_rad), np.sin(yaw_rad)]
                elif isinstance(robot_yaw_raw, (int, float)):
                    # It's a single number
                    yaw_rad = np.radians(robot_yaw_raw)  # Convert to radians
                    robot_orientation = [np.cos(yaw_rad), np.sin(yaw_rad)]

            # Global lidar pose (for transforming points)
            lidar_position = np.array(lidar.get("globalPosition", [0, 0, 0]))
            lidar_quaternion = np.array(lidar.get("globalRotation", [0, 0, 0, 1]))

            # Process annotations (actual lidar scan data)
            pointcloud = []
            for annot in lidar.get("annotations", []):
                ranges = np.array(annot.get("ranges", []))
                angles = np.array(annot.get("angles", []))
                classes = np.array(annot.get("object_classes", []))

                # Transform points to global frame
                if len(ranges) > 0 and len(angles) > 0:
                    pointcloud_x, pointcloud_y = transform_lidar_points(
                        ranges, angles, lidar_position, lidar_quaternion
                    )
                    
                    for i in range(len(pointcloud_x)):
                        # FIX: Handle 'no_class' string in class values
                        class_value = 0  # Default class
                        if i < len(classes):
                            try:
                                # Try to convert to int, but handle string value
                                if isinstance(classes[i], (int, float)):
                                    class_value = int(classes[i])
                                elif isinstance(classes[i], str) and classes[i].lower() != 'no_class':
                                    # Try to convert string to int if it's not 'no_class'
                                    class_value = int(classes[i])
                                # For 'no_class' or any other non-numeric string, keep default 0
                            except (ValueError, TypeError):
                                # Keep default class value if conversion fails
                                pass
                        
                        pointcloud.append({
                            "x": pointcloud_x[i],
                            "y": pointcloud_y[i],
                            "class": class_value
                        })

            # Append data to the structure
            robot_data[amr_id]["timestamps"].append(timestamp)
            robot_data[amr_id]["robot_positions"].append(robot_position)
            robot_data[amr_id]["robot_orientations"].append(robot_orientation)
            robot_data[amr_id]["pointclouds"].append(pointcloud)

    return robot_data


if __name__ == "__main__":
    main()