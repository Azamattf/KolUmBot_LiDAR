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
    sample_size = 100   # upper sampling bound, set to "" for all frames
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

import numpy as np
import cv2
from scipy.spatial.transform import Rotation
from tqdm import tqdm
import os

def create_lidar_overlay_images(json_files, save_path, path_to_images, class_def, point_size=3):
    """Create images with properly projected LiDAR points following perspective."""
    os.makedirs(os.path.join(save_path, "lidar_overlay"), exist_ok=True)
    
    # Sensor heights (in meters)
    LIDAR_HEIGHT = 0.14
    CAMERA_HEIGHT = 0.12
    HEIGHT_DIFFERENCE = LIDAR_HEIGHT - CAMERA_HEIGHT  # 0.02m

    # Step 1: Extract projection matrix
    global_proj_matrix = None
    for frame_data in json_files:
        cameras = [c for c in frame_data.get("captures", []) 
                 if c.get('@type', '').endswith('RGBCamera')]
        if cameras:
            global_proj_matrix = cameras[0].get("matrix")
            break
    
    if global_proj_matrix is None:
        raise ValueError("No camera found in JSON files!")

    # Step 2: Build camera matrix
    fx = global_proj_matrix[0]
    fy = global_proj_matrix[4]
    width, height = cameras[0].get("dimension", [1280, 720])
    cx, cy = width / 2, height / 2
    
    # Calculate vertical and horizontal FOV
    vertical_fov = 60  # Default Unity FOV (vertical)
    aspect_ratio = width / height
    horizontal_fov = 2 * np.arctan(np.tan(np.deg2rad(vertical_fov) / 2) * aspect_ratio)
    
    print(f"Horizontal FOV: {np.rad2deg(horizontal_fov):.2f} degrees")
    
    # Update focal lengths based on the vertical FOV (fx, fy)
    f_x = 0.5 * width / np.tan(np.deg2rad(vertical_fov / 2))
    f_y = 0.5 * height / np.tan(np.deg2rad(vertical_fov / 2))
    
    K = np.array([
        [f_x, 0, cx],
        [0, f_y, cy], 
        [0, 0, 1]
    ])

    print(f"\nK Matrix: {K}")

    for frame_idx, frame_data in enumerate(tqdm(json_files, desc="Processing frames")):
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
            
            overlay = img.copy()
            
            # Add timestamp
            timestamp = frame_data.get("timestamp", 0)
            cv2.putText(overlay, f"Time: {timestamp:.2f}s", (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
            
            # Get camera pose (3D)
            cam_pos = np.array(camera["globalPosition"])
            cam_rot = np.array(camera["globalRotation"])  # Quaternion (x,y,z,w)
            R_cam = Rotation.from_quat([cam_rot[0], cam_rot[1], cam_rot[2], cam_rot[3]]).as_matrix()
            
            # Process LiDAR points
            for lidar in lidars:
                lidar_pos = np.array(lidar["globalPosition"])
                lidar_rot = np.array(lidar["globalRotation"])
                R_lidar = Rotation.from_quat([lidar_rot[0], lidar_rot[1], lidar_rot[2], lidar_rot[3]]).as_matrix()
                
                for annot in lidar.get("annotations", []):
                    ranges = np.array(annot.get("ranges", []))
                    angles = np.array(annot.get("angles", []))
                    classes = annot.get("object_classes", ["unknown"]*len(ranges))
                    
                    if len(ranges) == 0:
                        continue
                    
                    # Convert to 3D points in horizontal plane (Y = height difference)
                    theta_rad = np.deg2rad(angles)
                    x_local = ranges * np.sin(theta_rad)
                    z_local = ranges * np.cos(theta_rad)
                    y_local = np.full_like(ranges, HEIGHT_DIFFERENCE)  # Constant height
                    
                    points_local = np.vstack((x_local, y_local, z_local)).T
                    
                    # Transform to world coordinates
                    points_world = (R_lidar @ points_local.T).T + lidar_pos
                    
                    # Transform to camera coordinates
                    points_cam = (R_cam.T @ (points_world - cam_pos).T).T
                    
                    # Project to 2D image coordinates (this creates perspective effect)
                    points_2d = (K @ points_cam.T).T
                    points_2d = points_2d[:, :2] / points_2d[:, 2:]  # Perspective divide
                    
                    # Filter valid points
                    valid = (
                        (points_2d[:, 0] >= 0) & (points_2d[:, 0] < width) & 
                        (points_2d[:, 1] >= 0) & (points_2d[:, 1] < height) & 
                        (points_cam[:, 2] > 0)  # Z > 0 (in front of camera)
                    )
                    points_2d = points_2d[valid]
                    classes = np.array(classes)[valid]
                    
                    # Draw points (they should now follow perspective)
                    for (x, y), cls in zip(points_2d, classes):
                        color = class_def.get(cls, 'rgba(255,0,0,1)')
                        color_rgb = tuple(int(x) for x in color.split('(')[1].split(')')[0].split(',')[:3])
                        cv2.circle(overlay, (int(x), int(y)), point_size, color_rgb, -1)
            
            # Add legend
            unique_classes = set()
            for lidar in lidars:
                for annot in lidar.get("annotations", []):
                    unique_classes.update(annot.get("object_classes", []))
            
            legend_x = width - 250
            legend_y = 40
            for i, cls in enumerate(sorted(unique_classes)):
                color = class_def.get(cls, 'rgba(255,0,0,1)')
                color_rgb = tuple(int(x) for x in color.split('(')[1].split(')')[0].split(',')[:3])
                cv2.rectangle(overlay, 
                            (int(legend_x), int(legend_y + i*30)), 
                            (int(legend_x + 20), int(legend_y + i*30 + 20)), 
                            color_rgb, -1)
                cv2.putText(overlay, cls, 
                           (int(legend_x + 25), int(legend_y + i*30 + 15)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
            
            # Final output
            output = cv2.addWeighted(overlay, 0.7, img, 0.3, 0)
            output_path = os.path.join(save_path, "lidar_overlay", f"lidar_overlay_{os.path.splitext(img_filename)[0]}.png")
            cv2.imwrite(output_path, output)
    print("\n✅ Image overlay complete!")

if __name__ == "__main__":
    main()