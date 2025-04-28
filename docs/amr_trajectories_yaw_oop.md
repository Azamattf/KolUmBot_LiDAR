# AMR Trajectories with Yaw Documentation

## Overview
Script that visualizes Autonomous Mobile Robot (AMR) trajectories with yaw direction in an interactive Plotly animation.

## Requirements
- Python libraries: os, json, csv, plotly, pandas, natsort, tqdm, numpy

## Usage
```python
# Set parameters in main()
solo_nr = 2  # dataset number
path_to_dataset = "path/to/dataset/solo_2"
frame_step = 40  # process every Nth frame

# Run script
python amr_trajectories_yaw_oop.py
```

## Function Overview

### `load_json_files(path_to_jsons, frame_step)`
Loads JSON files with sensor data using specified frame step.

### `unity_quaternion_to_ros_yaw(q_unity)`
Converts Unity quaternion to 2D heading direction vectors.

### `read_positions(data, savepath)`
Extracts robot positions and yaw from JSON data and saves to CSV.

## Classes

### `AMR`
Represents an Autonomous Mobile Robot with methods to:
- Track trajectory data
- Get position/orientation at specific frames
- Generate visualization traces

### `AMRAnimation`
Creates and manages the trajectory animation:
- Computes visual boundaries
- Generates arrow traces for orientation
- Creates interactive Plotly animation with time slider

## Workflow
1. Load JSON files with sensor data
2. Extract robot positions and orientation (yaw)
3. Create AMR objects for each robot with trajectory data
4. Generate interactive animation showing:
   - Robot positions as squares
   - Movement paths as lines with markers
   - Orientation as directional arrows
   - Time controls and slider

## Outputs
- CSV file: `{save_path}/AMR_positions.csv`
- Interactive HTML: `{save_path}/AMR_Animation_oop.html`
