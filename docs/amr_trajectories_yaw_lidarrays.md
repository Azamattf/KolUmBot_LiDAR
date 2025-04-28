# `amr_trajectories_yaw_lidarrays.py`

This script processes and visualizes Autonomous Mobile Robot (AMR) trajectories along with their LiDAR pointcloud data in an interactive HTML visualization.

## Features
- Processes JSON data from LiDAR scans and robot positions
- Applies temporal offset correction between robot poses and LiDAR data
- Transforms LiDAR points from local to global coordinate systems
- Generates an interactive HTML visualization with:
  - Robot trajectories and orientations
  - Color-coded LiDAR pointclouds
  - Timeline playback controls

## Configuration
Edit these parameters in `main()`:
- `path_to_dataset`: Path to the dataset folder
- `path_to_images`: Path to JSON files with LiDAR data
- `save_path`: Where to save the output files
- `sample_size`: Upper sampling bound (empty string for all frames)
- `frame_step`: Frame sampling interval
- `time_step`: Time between consecutive frames

## Key Functions
- `load_json_files()`: Loads and samples JSON files
- `read_data_with_offset()`: Extracts and processes robot and LiDAR data
- `transform_lidar_points()`: Transforms points from local to global coordinates
- `create_html_visualization()`: Generates interactive HTML visualization

## Visualization Features
- Timeline slider for frame navigation
- Play/pause and reset controls
- Adjustable playback speed
- Color-coded LiDAR points by semantic class
- Robot trajectories with orientation indicators
- Automatic scaling to fit all data

## Output
- Processed data saved as JSON
- Self-contained HTML visualization with Plotly.js
