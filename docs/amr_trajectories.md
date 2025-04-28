# amr_trajectories.py
This script visualizes the movement trajectories of Autonomous Mobile Robots (AMRs) by processing JSON files containing position data and creating an animated 2D plot in x-y plane converted from the Unity's x-z plane. The animation shows each robot's path over time with markers indicating their current positions.

## Key Features
* Processes JSON files containing robot position data
* Extracts and plots 2D trajectories (x-z coordinates from 3D positions)
* Creates smooth animations with traces showing complete paths, markers indicating current positions, and timestamp display
* Supports saving animations as MP4 videos
* Includes progress tracking during processing

## Input Requirements
- A directory containing JSON files from Unity simulation captures
- From each JSON file:
  - Timestamp information
  - Robot position data (globalPosition)
  - Robot identification (AMR IDs)

## Output
- matplotlib animation saved as an MP4 file
- Plot shows robot paths as colored lines
- Circular markers indicating current positions
- A legend identifying each robot
- A timestamp display in the upper left

## Functions
### `main()`

Workflow: Sets dataset path --> Calls `load_json_files()` to process JSON files --> Calls `read_positions()` to extract robot positions --> Calls `animate_robots()` to create and save animation

### `load_json_files(path_to_jsons, frame_step)`
- Purpose: Loads and samples JSON files
- Parameters:
  - `path_to_jsons`: Path to JSON files directory
  - `frame_step`: Sampling interval (process every Nth file)
- Returns:
  - List of loaded JSON data
- Features:
  - Natural sorting of files
  - Progress bar display
  - Error handling for malformed JSON

### `read_positions(data)`
- Purpose: Extracts position data from JSON data
- Returns:
  - Dictionary of robot positions: `{robot_id: [(timestamp, x, y), ...]}`
-  Features:
  - Extracts front lidar positions as robot locations
  - Converts 3D positions to 2D (x-z plane)
  - Handles missing position data

### `animate_robots(robot_positions, time_interval, save_path, frame_step)`
- Purpose: Creates trajectory animation
- Parameters:
  - `robot_positions`: Position data from read_positions()
  - `time_interval`: Delay between frames (seconds)
  - `save_path`
  - `frame_step`: Frame sampling, also displayed in title
- Features:
  - Auto-scaling plot limits
  - Color-coded robots
  - Continuous traces with position markers
  - Real-time timestamp display
  - Progress bar during video rendering
  - MP4 output with H.264 encoding

## Usage Instructions
Set the correct path_to_dataset for your environment

Adjust frame_step to control animation smoothness vs performance

Run the script: python amr_trajectories.py

The animation will display and optionally save to MP4

Customization Options
Change solo_nr to select different datasets

Adjust time_interval for playback speed

Modify colors in color_map

Change video encoding parameters in FFMpegWriter

Adjust figure size and DPI for output quality

## Dependencies
Python 3.x

Required packages:

os, json, glob, itertools

natsort, tqdm, numpy

matplotlib (with FFMpegWriter support)

IPython (for notebook display)



Performance Notes
Larger frame_step values will process faster but produce less smooth animations

Video rendering may take significant time for large datasets

Ensure FFmpeg is installed for MP4 output
