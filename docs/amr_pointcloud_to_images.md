# `amr_pointcloud_to_images.py`
This script overlays LiDAR point clouds onto camera images from Autonomous Mobile Robots (AMRs) and generates visualization videos.

## Requirements
- Python libraries: numpy, natsort, tqdm, scipy, opencv-python, PIL

## Usage
```python
# Set parameters in main()
path_to_dataset = "path/to/dataset"
path_to_images = "path/to/images"
sample_size = 1000  # max frames to process
frame_step = 10     # process every Nth frame

# Run script
python amr_pointcloud_to_images.py
```

## Functions

### `load_json_files(path_to_jsons, sample_size, frame_step)`
Loads JSON files containing sensor data with specified sampling rate.

### `load_class_definitions(filepath)`
Creates mapping from semantic class labels to RGBA colors.

### `create_lidar_overlay_images(json_files, save_path, overlay_folder_name, path_to_images, class_def)`
1. Projects 3D LiDAR points onto 2D camera images
2. Color-codes points by semantic class
3. Adds timestamp, AMR ID, and class legend

### `create_videos_per_amr(image_folder, output_folder, time_step, speed_factors)`
Creates videos for each AMR at different playback speeds.

## Workflow
1. Load JSON data files and semantic class definitions
2. Calculate camera projection matrix from FOV
3. Project LiDAR points to camera view using:
   - LiDAR local → World → Camera → Image coordinates
4. Draw points with class-specific colors on camera images
5. Group images by AMR and create videos at different speeds

## Outputs
- Overlay images: `{save_path}/lidar_overlay/lidar_overlay_{frame}.png`
- Videos: `{save_path}/videos/{amr_id}_speed_{speed_factor}x.mp4`
