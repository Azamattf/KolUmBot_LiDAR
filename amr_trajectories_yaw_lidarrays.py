import os
import json
import numpy as np
import plotly.graph_objects as go
from natsort import natsorted
from collections import defaultdict
from tqdm import tqdm
from scipy.spatial.transform import Rotation


def main():
    # Data paths
    path_to_dataset = "d:\\Works_Experience\\FML_Florian\\Kolumbot\\Lidar_visualization\\Unity_v3s\\"
    path_to_images = os.path.join(path_to_dataset, "sequence.0_json_only")
    save_path = os.path.join(path_to_dataset, "Export")
    path_to_sem_def = os.path.join(path_to_dataset, "class_definition_semantic_segmentation.json")
      
    # Processing parameters
    sample_size = 1900
    frame_step = 1

    # Load and process data
    json_files = load_json_files(path_to_images, sample_size, frame_step)
    #robot_data = read_data(json_files, save_path)
    robot_data = read_data_with_offset(json_files, save_path)
    class_def = load_class_definitions(path_to_sem_def)

    # Create visualization
    visualizer = Visualizer(robot_data, class_def)
    visualizer.animate()
    visualizer.save_animation(save_path)


def load_class_definitions(filepath):
    with open(filepath, 'r') as f:
        class_def = json.load(f)
    
    # Create color mapping from class labels to RGB values
    color_map = {}
    for entry in class_def.get("m_LabelEntries", []):
        label = entry.get("label", "unknown")
        r = int(entry.get("color", {}).get("r", 0) * 255)
        g = int(entry.get("color", {}).get("g", 0) * 255)
        b = int(entry.get("color", {}).get("b", 0) * 255)
        a = entry.get("color", {}).get("a", 1.0)
        color_map[label] = f'rgba({r},{g},{b},{a})'
    
    return color_map


class Visualizer:
    def __init__(self, robot_data, class_color_map):
        self.robot_data = robot_data
        self.class_color_map = class_color_map
        self.fig = go.Figure()
        
        # Define unique colors for each robot
        self.robot_colors = {
            robot_id: f'rgb({hash(robot_id) % 200},{(hash(robot_id) * 13) % 200},{(hash(robot_id) * 29) % 200})'
            for robot_id in robot_data.keys()
        }
        
        self.robot_shapes = {}  # Store robot shape traces
        self.robot_paths = {}   # Store robot path traces
        self.pointcloud_traces = {}  # Store pointcloud traces
        
        # Get common timestamps across all robots
        self.time_stamps = sorted(list(robot_data.values())[0]["timestamps"])
        
    def create_robot_shape(self, x, y, orientation, robot_id, size=2):
        
        # Calculate angle from orientation vector
        angle = np.arctan2(orientation[1], orientation[0])
        
        # Define trapezoid in local coordinates (centered at origin)
        front_width = size * 0.6
        back_width = size * 1.2
        length = size * 2
        
        local_points = np.array([
            [length/2, -front_width/2],  # front left
            [length/2, front_width/2],   # front right
            [-length/2, back_width/2],   # back right
            [-length/2, -back_width/2],  # back left
            [length/2, -front_width/2]   # close the shape
        ])
        
        # Create rotation matrix
        rotation = np.array([
            [np.sin(angle), -np.cos(angle)],
            [np.cos(angle), np.sin(angle)]
        ])
        
        # Apply rotation and translation to all points
        transformed_points = np.dot(local_points, rotation.T) + np.array([x, y])
        
        # Split into x and y coordinates
        points_x = transformed_points[:, 0]
        points_y = transformed_points[:, 1]
        
        return points_x, points_y

    def map_classes_to_colors(self, class_labels):
        """Map pointcloud classes to colors using the semantic segmentation definitions."""
        class_colors = []
        for label in class_labels:
            # Use the color map, default to gray if class not found
            color = self.class_color_map.get(label, 'rgba(150, 150, 150, 0.7)')
            class_colors.append(color)
        return class_colors
    
    def add_pointcloud_legend(self):
        """Add a legend showing the color mapping for semantic classes."""
        # Use the existing class_color_map directly
        for class_name, color in self.class_color_map.items():
            # Add an invisible scatter point with the class name and color
            self.fig.add_trace(go.Scatter(
                x=[None], 
                y=[None],
                mode='markers',
                marker=dict(
                    size=10,
                    color=color
                ),
                name=f"{class_name}",
                showlegend=True
            ))

    def animate(self):
        """Create animation with robot movements and LiDAR pointclouds."""
        # First, compute the bounds across ALL frames and robots
        all_x = []
        all_z = []
        
        for robot_id, data in self.robot_data.items():
            # Get all robot positions across all frames
            all_x.extend([pos[0] for pos in data["robot_positions"]])
            all_z.extend([pos[1] for pos in data["robot_positions"]])
            
            # Include pointcloud data from all frames
            for pointcloud in data["pointclouds"]:
                all_x.extend([p["x"] for p in pointcloud])
                all_z.extend([p["z"] for p in pointcloud])
        
        # 1. Calculate fixed BOUNDS

        useful_axis_part = 0.4
        x_min, x_max = min(all_x)*useful_axis_part, max(all_x)*useful_axis_part
        z_min, z_max = min(all_z)*useful_axis_part, max(all_z)*useful_axis_part

        x_axis_range = [x_min, x_max]
        z_axis_range = [z_min, z_max]
        
        # The proper ordering to ensure robots appear on top:
        # 1. First all pointcloud traces
        # 2. Then all path traces
        # 3. Finally all robot shape traces
        
        # 2. INITIALIZE

        # Initialize pointcloud traces first
        for robot_id, data in self.robot_data.items():
            # Initial pointcloud data
            pointclouds = data["pointclouds"][0]
            pointcloud_x = [p["x"] for p in pointclouds]
            pointcloud_z = [p["z"] for p in pointclouds]
            pointcloud_classes = [p.get("class", "unknown") for p in pointclouds]
            colors = self.map_classes_to_colors(pointcloud_classes)
            
            # Add pointcloud trace
            self.pointcloud_traces[robot_id] = go.Scatter(
                x=pointcloud_x,
                y=pointcloud_z,
                mode="markers",
                marker=dict(
                    color=colors,
                    size=4,
                    opacity=0.7
                ),
                name=f"{robot_id} (LiDAR)",
                showlegend=False
            )
            self.fig.add_trace(self.pointcloud_traces[robot_id])
        
        # Then initialize path traces
        for robot_id, data in self.robot_data.items():
            robot_color = self.robot_colors[robot_id]
            initial_pos = data["robot_positions"][0]
            
            # Add robot path trace (initially empty)
            self.robot_paths[robot_id] = go.Scatter(
                x=[initial_pos[0]],
                y=[initial_pos[1]],
                mode="lines",
                line=dict(color=robot_color, width=2),
                name=f"{robot_id} (Path)",
                showlegend=False
            )
            self.fig.add_trace(self.robot_paths[robot_id])
        
        # Finally initialize robot shape traces
        for robot_id, data in self.robot_data.items():
            robot_color = self.robot_colors[robot_id]
            
            # Initial position and orientation
            initial_pos = data["robot_positions"][0]
            initial_orient = data["robot_orientations"][0]
            
            # Create robot shape (trapezoid)
            shape_x, shape_y = self.create_robot_shape(
                initial_pos[0], initial_pos[1], initial_orient, robot_id
            )
            
            # Add robot shape trace
            self.robot_shapes[robot_id] = go.Scatter(
                x=shape_x,
                y=shape_y,
                mode="lines",
                fill="toself",
                fillcolor=robot_color,  # Ensure matching color with path
                line=dict(color=robot_color, width=1),
                name=f"{robot_id}",
                text=robot_id,
                hoverinfo="text"
            )
            self.fig.add_trace(self.robot_shapes[robot_id])
        
        # 3. Create FRAMES for animation
        frames = []
        for frame_idx, timestamp in tqdm(enumerate(self.time_stamps), total=len(self.time_stamps), ascii="░▒█", desc="Processing frames", unit=" frames"):
            frame_data = []
            
            # Follow the same ordering in each frame: pointclouds first, then paths, then robots
            # First add all pointcloud traces for this frame
            for robot_id, data in self.robot_data.items():
                # Update pointcloud
                pointclouds = data["pointclouds"][frame_idx]
                pointcloud_x = [p["x"] for p in pointclouds]
                pointcloud_z = [p["z"] for p in pointclouds]
                pointcloud_classes = [p.get("class", "unknown") for p in pointclouds]
                colors = self.map_classes_to_colors(pointcloud_classes)
                
                frame_data.append(
                    go.Scatter(
                        x=pointcloud_x,
                        y=pointcloud_z,
                        mode="markers",
                        marker=dict(
                            color=colors,
                            size=4,
                            opacity=0.7
                        ),
                        name=f"{robot_id} (LiDAR)",
                        showlegend=False
                    )
                )
            
            # Add all path traces
            for robot_id, data in self.robot_data.items():
                robot_color = self.robot_colors[robot_id]
                
                # Collect all positions up to current frame for path
                path_x = [pos[0] for pos in data["robot_positions"][:frame_idx+1]]
                path_y = [pos[1] for pos in data["robot_positions"][:frame_idx+1]]
                
                frame_data.append(
                    go.Scatter(
                        x=path_x,
                        y=path_y,
                        mode="lines",
                        line=dict(color=robot_color, width=2),
                        name=f"{robot_id} (Path)",
                        showlegend=False
                    )
                )
            
            # Finally add all robot shapes
            for robot_id, data in self.robot_data.items():
                robot_color = self.robot_colors[robot_id]
                
                # Robot position and orientation for this frame
                robot_pos = data["robot_positions"][frame_idx]
                robot_orient = data["robot_orientations"][frame_idx]
                
                # Update robot shape
                shape_x, shape_y = self.create_robot_shape(
                    robot_pos[0], robot_pos[1], robot_orient, robot_id
                )
                
                frame_data.append(
                    go.Scatter(
                        x=shape_x,
                        y=shape_y,
                        mode="lines",
                        fill="toself",
                        fillcolor=robot_color,  # Use same color as the path
                        line=dict(color=robot_color, width=1),
                        name=f"{robot_id}",
                        text=f"{robot_id} @ {timestamp}",
                        hoverinfo="text"
                    )
                )
            
            # Create frame with all data - ENSURE FIXED AXES in every frame
            frame = go.Frame(
                data=frame_data,
                name=str(f"{timestamp:.2f}"),
                layout=go.Layout(
                    xaxis=dict(range=x_axis_range, fixedrange=True),  # Add fixedrange=True
                    yaxis=dict(range=z_axis_range, fixedrange=True)   # Add fixedrange=True
                )
            )
            frames.append(frame)
        
        # 4. Add frames to figure
        self.fig.frames = frames

        self.add_pointcloud_legend()
        
        # 5. Add animation controls
        self.fig.update_layout(
            updatemenus=[{
                'type': 'buttons',
                'buttons': [
                    {
                        'args': [None, {'frame': {'duration': 50, 'redraw': True}, 'fromcurrent': True, 'mode': 'immediate', 'transition': {'duration': 0}}],
                        'args2': [[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate', 'transition': {'duration': 0}}],
                        'label': 'Play/Pause',
                        'method': 'animate'
                    }
                ],
                'direction': 'left',
                'pad': {'r': 10, 't': 10},
                'showactive': False,
                'type': 'buttons',
                'x': 0.1,
                'xanchor': 'right',
                'y': 0,
                'yanchor': 'top'
            }],

            sliders=[{
                'active': 0,
                'steps': [{
                    'args': [[f.name], {'frame': {'duration': 50, 'redraw': True}, 'mode': 'immediate'}],
                    'label': f.name,
                    'method': 'animate'
                } for f in frames],
                'x': 0.1,
                'xanchor': 'left',
                'y': 0,
                'yanchor': 'top'
            }]
        )
        
        # 6. Set up plot for Unity coordinates with fixed axis ranges
        self.fig.update_layout(
            title="AMR Movement with LiDAR Data (Unity Coordinates)",
            xaxis=dict(
                title="X Position (Unity)",
                range=x_axis_range,  # Fixed range 
                scaleanchor="y",
                scaleratio=1,
                fixedrange=True,     # prevent zoom
                constrain="domain"   # maintain aspect ratio
            ),
            yaxis=dict(
                title="Z Position (Unity)",  # Z is up in Unity
                range=z_axis_range,  
                fixedrange=True,     
                constrain="domain"  
            ),
            legend=dict(
                x=1.05,
                y=1,
                xanchor='left',
                yanchor='top'
            ),
            margin=dict(l=20, r=20, t=30, b=20),
            plot_bgcolor='rgba(240, 240, 240, 0.8)',
            height=800,
            width=1000,
            scene_aspectmode='data',  # Preserve aspect ratio
            dragmode=False            # Disable dragging to prevent repositioning
        )

    def save_animation(self, savepath):
        """Save the animation to an HTML file."""
        os.makedirs(savepath, exist_ok=True)
        filepath = os.path.join(savepath, "AMR_Animation_with_LiDAR_Tracking.html")
        self.fig.write_html(filepath, auto_play=False)
        print(f"\n✅ Animation saved to {filepath}")


def transform_lidar_points(
    ranges: np.ndarray,
    angles: np.ndarray,
    lidar_position: np.ndarray,
    lidar_quaternion: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Transform lidar points from local to global Unity coordinate system."""
    # 1. Convert polar to local Cartesian coordinates (in Unity coordinates)
    theta_rad = np.deg2rad(angles)
    x_local = ranges * np.sin(theta_rad)
    z_local = ranges * np.cos(theta_rad)
    
    # Stack coordinates and add y=0 for 3D rotation (Unity uses XYZ where Y is up)
    points_local = np.vstack((x_local, np.zeros_like(x_local), z_local)).T
    
    # 2. Create rotation from quaternion
    R = Rotation.from_quat(lidar_quaternion)
    
    # 3. Apply rotation
    points_rotated = R.apply(points_local)
    
    # 4. Apply translation
    x_global = points_rotated[:, 0] + lidar_position[0]
    z_global = points_rotated[:, 2] + lidar_position[2]
    
    return x_global, z_global


def load_json_files(path_to_jsons, sample_size, frame_step):
    """Load JSON files from the specified directory with given frame step."""
    data = []
    json_files = [f for f in natsorted(os.listdir(path_to_jsons)) if f.endswith(".json")][1000:sample_size]
    sampled_files = json_files[::frame_step]
    
    print("\nFirst 5 sampled files:")
    print(f"{sampled_files[:5]} ...")

    for i, filename in tqdm(list(enumerate(sampled_files)), ascii="░▒█", desc="Loading JSON files", unit=" file"):
        file_path = os.path.join(path_to_jsons, filename)

        with open(file_path, "r", encoding='utf-8') as f:
            try:
                data.append(json.load(f))
            except json.JSONDecodeError:
                print(f"Error decoding {filename}, skipping.")
                continue

    print(f"\nLoaded {len(data)} frames (Step = {frame_step})")
                                
    return data


def read_data(data, savepath, filename="robot_lidar_data.json"):
    """Extract robot positions, orientations, and pointcloud data from JSON files.
    
    Note: Using Unity coordinate system where X-Z is the floor plane and Y is up.
    """
    robot_data = defaultdict(lambda: {
        "timestamps": [],
        "robot_positions": [],
        "robot_orientations": [],
        "pointclouds": []
    })

    for frame_index, frame in tqdm(list(enumerate(data)), ascii="░▒█", desc="Processing JSON files", unit="frame"):
        timestamp = frame.get("timestamp", frame_index)  # Use frame index if timestamp not available

        # Filter out 2D lidars
        lidars = [o for o in frame.get("captures", []) if o.get('@type') == 'type.custom/solo.2DLidar']

        # Track which robots we've processed in this frame
        processed_robots = set()

        for lidar in lidars:
            amr_id = "_".join(lidar["id"].split("_")[:2])  # e.g., "AMR_1"
            
            # Skip if we've already processed this robot in the current frame
            if amr_id in processed_robots:
                continue
            
            processed_robots.add(amr_id)

            # Get robot position and orientation
            robot_position_raw = lidar.get("robotPosition", [0, 0, 0])
            robot_yaw_raw = lidar.get("robotRotationEuler", [0, 0, 0])  # In radians
            
            # Take X and Z only (floor plan view in Unity)
            robot_position = [robot_position_raw[0], robot_position_raw[2]]
            
            # In Unity, y-axis rotation gives the yaw in the x-z plane
            # The rotation around y-axis corresponds to the angle in the x-z plane
            robot_orientation = [np.cos(robot_yaw_raw), np.sin(robot_yaw_raw)]

            # Add robot data for this frame
            robot_data[amr_id]["timestamps"].append(timestamp)
            robot_data[amr_id]["robot_positions"].append(robot_position)
            robot_data[amr_id]["robot_orientations"].append(robot_orientation)
            
            # Process all pointclouds for this robot
            all_points = []
            
            # Loop through all lidars for this robot to collect point cloud data
            robot_lidars = [l for l in lidars if "_".join(l["id"].split("_")[:2]) == amr_id]
            
            for robot_lidar in robot_lidars:
                # Global lidar pose
                lidar_position = np.array(robot_lidar.get("globalPosition", [0, 0, 0]))
                lidar_quaternion = np.array(robot_lidar.get("globalRotation", [0, 0, 0, 1]))
                
                # Process annotations (lidar scan data)
                for annot in robot_lidar.get("annotations", []):
                    ranges = np.array(annot.get("ranges", []))
                    angles = np.array(annot.get("angles", []))
                    classes = annot.get("object_classes", ["unknown"] * len(ranges))
                    
                    # Skip if no data
                    if len(ranges) == 0 or len(angles) == 0:
                        continue
                    
                    # Transform points to global frame (Unity coordinates)
                    pointcloud_x, pointcloud_z = transform_lidar_points(
                        ranges, angles, lidar_position, lidar_quaternion
                    )
                    
                    # Add transformed points to the collection
                    for x, z, c in zip(pointcloud_x, pointcloud_z, classes):
                        all_points.append({"x": float(x), "z": float(z), "class": c})
            
            # Add pointcloud data for this robot in this frame
            robot_data[amr_id]["pointclouds"].append(all_points)
    
    # Save data as JSON
    os.makedirs(savepath, exist_ok=True)
    filepath = os.path.join(savepath, filename)
    
    # Convert to regular dict for JSON serialization
    robot_data_dict = {k: dict(v) for k, v in robot_data.items()}
    
    with open(filepath, 'w') as f:
        json.dump(robot_data_dict, f, indent=2)
    
    print(f"\n✅ Robot and LiDAR data saved to {filepath}")
    
    return robot_data


def read_data_with_offset(data, savepath, offset=0.1*(1/3), filename="robot_lidar_data_transform_offset.json"):
    """Extract robot positions, orientations, and pointcloud data from JSON files,
    applying a temporal offset between lidar pose and pointcloud data.
    
    The offset is applied by using the current pointcloud data but transforming it with 
    the lidar pose from a different timestamp, correctly preserving fixed world objects.
    
    Args:
        data: The original data to process
        savepath: Directory to save the processed data
        offset: Temporal offset in seconds (default: 0.5)
        filename: Output filename (default: "robot_lidar_data_transform_offset.json")
    
    Returns:
        Dictionary containing the processed robot data with temporal offset correction
    """
    robot_data = defaultdict(lambda: {
        "timestamps": [],
        "robot_positions": [],
        "robot_orientations": [],
        "pointclouds": []
    })

    # First pass: Collect all frames with their timestamps
    all_frames = []
    for frame_index, frame in tqdm(list(enumerate(data)), ascii="░▒█", desc="Collecting frame data", unit="frame"):
        timestamp = frame.get("timestamp", frame_index)
        all_frames.append({
            "frame": frame,
            "timestamp": timestamp,
            "index": frame_index
        })
    
    # Sort frames by timestamp to ensure proper ordering
    all_frames.sort(key=lambda x: x["timestamp"])
    
    # Process each frame with the correct temporal offset
    for frame_data in tqdm(all_frames, ascii="░▒█", desc="Processing frames with offset", unit="frame"):
        frame = frame_data["frame"]
        current_timestamp = frame_data["timestamp"]
        
        # Filter out 2D lidars in current frame
        lidars = [o for o in frame.get("captures", []) if o.get('@type') == 'type.custom/solo.2DLidar']

        # Track which robots we've processed in this frame
        processed_robots = set()

        for lidar in lidars:
            amr_id = "_".join(lidar["id"].split("_")[:2])  # e.g., "AMR_1"
            
            # Skip if we've already processed this robot in the current frame
            if amr_id in processed_robots:
                continue
            
            processed_robots.add(amr_id)
            
            # Get current robot position and orientation (this will be recorded as-is)
            robot_position_raw = lidar.get("robotPosition", [0, 0, 0])
            robot_yaw_raw = lidar.get("robotRotationEuler", 0)  # In radians
            
            # Take X and Z only (floor plan view in Unity)
            robot_position = [robot_position_raw[0], robot_position_raw[2]]
            
            # In Unity, y-axis rotation gives the yaw in the x-z plane
            robot_orientation = [np.cos(robot_yaw_raw), np.sin(robot_yaw_raw)]

            # Find frame that corresponds to the offset timestamp
            offset_timestamp = current_timestamp - offset
            
            # Find the frame with the closest timestamp
            closest_frame = min(all_frames, key=lambda x: abs(x["timestamp"] - offset_timestamp), default=None)
            
            if closest_frame is None:
                # If no offset frame found, skip this robot for this frame
                continue
                
            offset_frame = closest_frame["frame"]
            
            # Find this robot's lidar in the offset frame
            offset_lidars = [o for o in offset_frame.get("captures", []) 
                            if o.get('@type') == 'type.custom/solo.2DLidar' 
                            and "_".join(o["id"].split("_")[:2]) == amr_id]
            
            if not offset_lidars:
                # If robot not found in offset frame, skip this robot for this frame
                continue
                
            # Add current frame data to result
            robot_data[amr_id]["timestamps"].append(current_timestamp)
            robot_data[amr_id]["robot_positions"].append(robot_position)
            robot_data[amr_id]["robot_orientations"].append(robot_orientation)
            
            # Process all pointclouds for this robot using CURRENT frame but OFFSET lidar poses
            all_points = []
            
            # Loop through all lidars for this robot in CURRENT frame
            robot_lidars = [l for l in lidars if "_".join(l["id"].split("_")[:2]) == amr_id]
            
            # Match current lidars with offset lidars by ID
            for robot_lidar in robot_lidars:
                lidar_id = robot_lidar["id"]
                
                # Find the matching lidar in the offset frame
                matching_offset_lidars = [l for l in offset_lidars if l["id"] == lidar_id]
                
                if not matching_offset_lidars:
                    continue
                    
                offset_lidar = matching_offset_lidars[0]
                
                # Get the global lidar pose from the OFFSET frame
                offset_lidar_position = np.array(offset_lidar.get("globalPosition", [0, 0, 0]))
                offset_lidar_rotation = np.array(offset_lidar.get("globalRotation", [0, 0, 0, 1]))
                
                # Process annotations (lidar scan data) from CURRENT frame
                for annot in robot_lidar.get("annotations", []):
                    ranges = np.array(annot.get("ranges", []))
                    angles = np.array(annot.get("angles", []))
                    classes = annot.get("object_classes", ["unknown"] * len(ranges))
                    
                    # Skip if no data
                    if len(ranges) == 0 or len(angles) == 0:
                        continue
                    
                    # Transform current points using offset lidar pose
                    pointcloud_x, pointcloud_z = transform_lidar_points(
                        ranges, angles, offset_lidar_position, offset_lidar_rotation
                    )
                    
                    # Add transformed points to the collection
                    for x, z, c in zip(pointcloud_x, pointcloud_z, classes):
                        all_points.append({"x": float(x), "z": float(z), "class": c})
            
            # Add pointcloud data for this robot in this frame
            robot_data[amr_id]["pointclouds"].append(all_points)
    
    # Save data as JSON
    os.makedirs(savepath, exist_ok=True)
    filepath = os.path.join(savepath, filename)
    
    # Convert to regular dict for JSON serialization
    robot_data_dict = {k: dict(v) for k, v in robot_data.items()}
    
    with open(filepath, 'w') as f:
        json.dump(robot_data_dict, f, indent=2)
    
    print(f"\n✅ Robot and LiDAR data with {offset}s transform offset saved to {filepath}")
    
    return robot_data

if __name__ == "__main__":
    main()