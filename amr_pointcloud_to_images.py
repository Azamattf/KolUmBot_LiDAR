import os
import json
import numpy as np
from natsort import natsorted
from collections import defaultdict
from tqdm import tqdm
from scipy.spatial.transform import Rotation
import cv2
from PIL import Image


def main():
    # Data paths
    path_to_dataset = "d:\\Works_Experience\\FML_Florian\\Kolumbot\\Lidar_visualization\\Unity_v3_full\\"
    path_to_images = os.path.join(path_to_dataset, "sequence.0")
    save_path = os.path.join(path_to_dataset, "Export")
    path_to_sem_def = os.path.join(path_to_dataset, "semantic_segmentation_definition.json")
      
    # Processing parameters
    sample_size = 1000   # upper sampling bound, set to "" for all frames
    frame_step = 10
    time_step = 0.0333333351*frame_step
    
    # Load and process data
    json_files = load_json_files(path_to_images, sample_size, frame_step)
    #robot_data = read_data_with_offset(json_files, save_path)
    class_def = load_class_definitions(path_to_sem_def)

    # Process each frame to create pointcloud-overlaid images
    create_lidar_overlay_images(json_files, save_path, path_to_images, class_def)


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
    json_files = [f for f in natsorted(os.listdir(path_to_jsons)) if f.endswith(".json")]
    sample_size = int(sample_size) if str(sample_size).isdigit() else len(json_files)
    sampled_files = json_files[:sample_size][::frame_step]
    
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

def read_data_with_offset(data, savepath, offset=0.3*(1/3), filename="robot_lidar_data_transform_offset.json"):
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

def calculate_bounds(robot_data):
    """Calculate bounds for all frames across all robots."""
    all_x = []
    all_z = []
    
    for robot_id, data in robot_data.items():
        # Get all robot positions across all frames
        all_x.extend([pos[0] for pos in data["robot_positions"]])
        all_z.extend([pos[1] for pos in data["robot_positions"]])
        
        # Include pointcloud data from all frames
        for pointcloud in data["pointclouds"]:
            all_x.extend([p["x"] for p in pointcloud])
            all_z.extend([p["z"] for p in pointcloud])
    
    # Calculate the view bounds
    useful_axis_part = 0.5
    x_min, x_max = min(all_x)*useful_axis_part, max(all_x)*useful_axis_part*0.8
    z_min, z_max = min(all_z)*useful_axis_part*0.8, max(all_z)*useful_axis_part
    
    return [x_min, x_max], [z_min, z_max]

def create_lidar_overlay_images(json_files, save_path, path_to_images, class_def, point_size=3):
    """Create images with LiDAR pointclouds overlaid on camera views, including timestamp and legend."""
    os.makedirs(os.path.join(save_path, "lidar_overlay"), exist_ok=True)
    
    # Step 1: Extract the projection matrix from the FIRST camera
    global_proj_matrix = None
    for frame_data in json_files:
        cameras = [c for c in frame_data.get("captures", []) 
                  if c.get('@type', '').endswith('RGBCamera')]
        if cameras:
            global_proj_matrix = cameras[0].get("matrix")
            break
    
    if global_proj_matrix is None:
        raise ValueError(f"No camera found in JSON files to extract projection matrix!")
    
    # Step 2: Build the intrinsic matrix (K)
    fx = global_proj_matrix[0]
    fy = global_proj_matrix[4]
    width, height = cameras[0].get("dimension", [1280, 720])
    cx, cy = width / 2, height / 2
    
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])

    # Step 3: Process all frames
    for frame_idx, frame_data in enumerate(tqdm(json_files, desc="Creating overlay images")):
        cameras = [c for c in frame_data.get("captures", []) 
                  if c.get('@type', '').endswith('RGBCamera')]
        lidars = [l for l in frame_data.get("captures", []) 
                 if l.get('@type') == 'type.custom/solo.2DLidar']
        
        if not cameras or not lidars:
            continue

        for camera in cameras:
            img_filename = camera.get("filename", "")
            if not img_filename:
                continue
            
            img_path = os.path.join(path_to_images, img_filename)
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            # Create overlay image
            overlay = img.copy()
            
            # Get timestamp and format it
            timestamp = frame_data.get("timestamp", 0)
            readable_time = f"Time: {timestamp:.2f}s"
            
            # Add timestamp to image (top-left corner)
            cv2.putText(overlay, readable_time, (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Prepare legend data
            unique_classes = set()
            for lidar in lidars:
                for annot in lidar.get("annotations", []):
                    classes = annot.get("object_classes", [])
                    unique_classes.update(classes)
            
            # Add legend (top-right corner)
            legend_x = width - 250
            legend_y = 40
            for i, cls in enumerate(sorted(unique_classes)):
                color = class_def.get(cls, 'rgba(255,0,0,1)')
                color_rgb = tuple(int(x) for x in color.split('(')[1].split(')')[0].split(',')[:3])
                
                # Convert all coordinates to integers
                pt1 = (int(legend_x), int(legend_y + i*30 - 15))
                pt2 = (int(legend_x + 20), int(legend_y + i*30 + 5))
                
                # Draw legend color box
                cv2.rectangle(overlay, pt1, pt2, color_rgb, -1)
                
                # Draw class label
                cv2.putText(overlay, cls, (int(legend_x + 30), int(legend_y + i*30)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

            # Process LiDAR points (same as before)
            cam_pos = np.array(camera["globalPosition"])
            cam_rot = np.array(camera["globalRotation"])
            R_cam = Rotation.from_quat([cam_rot[0], cam_rot[1], cam_rot[2], cam_rot[3]]).as_matrix()
            
            for lidar in lidars:
                lidar_pos = np.array(lidar["globalPosition"])
                lidar_rot = np.array(lidar["globalRotation"])
                R_lidar = Rotation.from_quat([lidar_rot[0], lidar_rot[1], lidar_rot[2], lidar_rot[3]]).as_matrix()
                
                for annot in lidar.get("annotations", []):
                    ranges = np.array(annot.get("ranges", []))
                    angles = np.array(annot.get("angles", []))
                    classes = annot.get("object_classes", ["unknown"] * len(ranges))
                    
                    if len(ranges) == 0:
                        continue
                    
                    x_local = ranges * np.sin(np.deg2rad(angles))
                    z_local = ranges * np.cos(np.deg2rad(angles))
                    points_local = np.vstack((x_local, np.zeros_like(x_local), z_local)).T
                    points_world = (R_lidar @ points_local.T).T + lidar_pos
                    points_cam = (R_cam.T @ (points_world - cam_pos).T).T
                    points_2d = (K @ points_cam.T).T
                    points_2d = points_2d[:, :2] / points_2d[:, 2:]
                    
                    valid = (
                        (points_2d[:, 0] >= 0) & (points_2d[:, 0] < width) &
                        (points_2d[:, 1] >= 0) & (points_2d[:, 1] < height) &
                        (points_cam[:, 2] > 0)
                    )
                    points_2d = points_2d[valid]
                    classes = np.array(classes)[valid]
                    
                    for (x, y), cls in zip(points_2d, classes):
                        color = class_def.get(cls, 'rgba(255,0,0,1)')
                        color_rgb = tuple(int(x) for x in color.split('(')[1].split(')')[0].split(',')[:3])
                        cv2.circle(overlay, (int(x), int(y)), point_size, color_rgb, -1)
            
            # Final output with transparency
            output = cv2.addWeighted(overlay, 0.7, img, 0.3, 0)
            output_path = os.path.join(save_path, "lidar_overlay", f"lidar_overlay_{os.path.splitext(img_filename)[0]}.png")
            cv2.imwrite(output_path, output)
        print("\n✅ Image overlay complete!")

if __name__ == "__main__":
    main()