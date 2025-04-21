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
    lidar_overlay_folder_name = "lidar_overlay"
      
    # Processing parameters
    sample_size = 1000  # upper sampling bound, set to "" for all frames
    frame_step = 10
    time_step = 0.0333333351*frame_step
    
    # Load and process data
    json_files = load_json_files(path_to_images, sample_size, frame_step)
    class_def = load_class_definitions(path_to_sem_def)
    
    # Process each frame to create pointcloud-overlaid images
    create_lidar_overlay_images(json_files, save_path, lidar_overlay_folder_name, path_to_images, class_def)

    # Create overlay videos for each AMR
    create_videos_per_amr(os.path.join(save_path, lidar_overlay_folder_name), os.path.join(save_path, "videos"), speed_factors=[1.0, 2.0])


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

def create_lidar_overlay_images(json_files, save_path, overlay_folder_name, path_to_images, class_def, point_size=3):
    """Create images with properly projected LiDAR points following perspective."""
    os.makedirs(os.path.join(save_path, "lidar_overlay"), exist_ok=True)
    
    # Sensor heights (in meters)
    LIDAR_HEIGHT = 0.14
    """ CAMERA_HEIGHT = 0.12; this is not needed since globalPosition data already contains this information"""

    # Step 1: Extract camera information
    global_proj_matrix = None
    width, height = 1280, 720  # Default
    
    for frame_data in json_files:
        cameras = [c for c in frame_data.get("captures", []) 
                 if c.get('@type', '').endswith('RGBCamera')]
        if cameras:
            global_proj_matrix = cameras[0].get("matrix")
            width, height = cameras[0].get("dimension", [1280, 720])
            break
    
    if global_proj_matrix is None:
        raise ValueError("No camera found in JSON files!")

    # Step 2: Using vertical FOV = 60 degrees to build camera matrix
    vertical_fov = 60  # in degrees
    aspect_ratio = width / height
    horizontal_fov = 2 * np.arctan(np.tan(np.deg2rad(vertical_fov) / 2) * aspect_ratio)
    
    print(f"Vertical FOV: {vertical_fov:.2f} degrees")
    print(f"Horizontal FOV: {np.rad2deg(horizontal_fov):.2f} degrees")
    
    # Calculate focal lengths based on the vertical FOV
    f_y = height / (2 * np.tan(np.deg2rad(vertical_fov) / 2))
    f_x = f_y  # For square pixels, focal length is the same in x and y when measured in pixels
    
    # Principal point (usually the center of the image)
    cx = width / 2
    cy = height / 2
    
    K = np.array([
        [f_x, 0, cx],
        [0, -f_y, cy], 
        [0, 0, 1]
    ])

    print(f"\nCamera Intrinsic Matrix K:\n{K}")

    # Step3: Frame processing
    for frame_idx, frame_data in enumerate(tqdm(json_files, desc="Processing frames", unit="frame")):
        cameras = [c for c in frame_data.get("captures", []) 
                  if c.get('@type', '').endswith('RGBCamera')]
        lidars = [l for l in frame_data.get("captures", [])
                 if l.get('@type') == 'type.custom/solo.2DLidar']
        
        if not cameras or not lidars:
            continue

        for camera in cameras:
        
            camera_id = camera.get("id", "")
            amr_id = f"AMR_{camera_id.split("_")[1]}"
            
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
            cv2.putText(overlay, f"AMR: {amr_id}", (20, 70), 
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
                    
                    # Convert to 3D points in horizontal plane (Y = lidar height)
                    theta_rad = np.deg2rad(angles)
                    x_local = ranges * np.sin(theta_rad)
                    z_local = ranges * np.cos(theta_rad)
                    y_local = np.full_like(ranges, LIDAR_HEIGHT)  # Constant height
                    
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
            output_path = os.path.join(save_path, overlay_folder_name, f"lidar_overlay_{os.path.splitext(img_filename)[0]}.png")
            cv2.imwrite(output_path, output)
    print("\nImage overlay complete!")

def create_videos_per_amr(image_folder, output_folder, fps=30, speed_factors=[1.0, 1.25, 1.5, 2.0]):
    """Create videos at different playback speeds for each AMR.
    
    Args:
        image_folder: Path to folder containing overlay images
        output_folder: Path to save output videos
        fps: Base frames per second
        speed_factors: List of playback speed multipliers (1x, 1.25x, etc.)
    """
    os.makedirs(output_folder, exist_ok=True)
    amr_groups = defaultdict(list)

    # Group images by AMR ID
    for filename in os.listdir(image_folder):
        if not filename.endswith(".png") or not filename.startswith("lidar_overlay_"):
            continue
            
        # Extract AMR ID from filename (format: lidar_overlay_step120.AMR_3_camera)
        parts = filename.split('.')
        if len(parts) < 2:
            continue
            
        amr_part = parts[1]  # This should be "AMR_3_camera" in your example
        if amr_part.startswith("AMR_"):
            # Extract just the AMR ID (e.g., "AMR_3")
            amr_id = amr_part.split('_')[:2]
            amr_id = '_'.join(amr_id)  # Joins ["AMR", "3"] to "AMR_3"
            amr_groups[amr_id].append(filename)
        
    print(f"Found {len(amr_groups)} AMRs: {list(amr_groups.keys())}")

    # Create video for each AMR at different speeds
    for amr_id, files in amr_groups.items():
        # Natural sort to ensure correct frame order
        files = natsorted(files)
        if not files:
            continue

        print(f"Processing AMR '{amr_id}' with {len(files)} frames")
        
        first_image_path = os.path.join(image_folder, files[0])
        frame = cv2.imread(first_image_path)
        if frame is None:
            print(f"Warning: Could not read image {first_image_path}")
            continue
            
        height, width, _ = frame.shape

        # Create videos at different speeds
        for speed_factor in speed_factors:
            # Adjust FPS for different playback speeds
            effective_fps = fps * speed_factor
            
            video_path = os.path.join(output_folder, f"{amr_id}_speed_{speed_factor:.2f}x.mp4")
            out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), effective_fps, (width, height))

            print(f"Creating video for {amr_id} at {speed_factor:.2f}x speed with {len(files)} frames...")
            
            for fname in tqdm(files, desc=f"Writing {amr_id} at {speed_factor:.2f}x"):
                img = cv2.imread(os.path.join(image_folder, fname))
                if img is None:
                    print(f"Warning: Could not read image {fname}")
                    continue
                    
                # Add speed indicator to the frame
                speed_text = f"Playback: {speed_factor:.2f}x"
                cv2.putText(img, speed_text, (width - 200, height - 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                
                out.write(img)

            out.release()
            print(f"Saved: {video_path}")

if __name__ == "__main__":
    main()