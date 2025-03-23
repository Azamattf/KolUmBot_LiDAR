import os
import json
import numpy as np
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
    sample_size = ""   # upper sampling bound, set to "" for all frames
    frame_step = 10
    time_step = 0.033333335*frame_step
    
    # Load and process data
    json_files = load_json_files(path_to_images, sample_size, frame_step)
    robot_data = read_data_with_offset(json_files, save_path)
    class_def = load_class_definitions(path_to_sem_def)
    
    # Create and save HTML
    create_html_visualization(robot_data, class_def, frame_step, time_step, save_path)


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
    sample_size = int(sample_size) if str(sample_size).isdigit() else len(json_files)
    json_files = [f for f in natsorted(os.listdir(path_to_jsons)) if f.endswith(".json")][:sample_size]
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


def create_robot_shape(x, y, orientation, size=2):
    """Create a robot shape (trapezoid) at the given position and orientation."""
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
    points_x = transformed_points[:, 0].tolist()
    points_y = transformed_points[:, 1].tolist()
    
    return points_x, points_y


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


def prepare_visualization_data(robot_data, class_color_map):
    """Prepare visualization data for Plotly."""
    # Define unique colors for each robot
    robot_colors = {
        robot_id: f'rgb({hash(robot_id) % 200},{(hash(robot_id) * 13) % 200},{(hash(robot_id) * 29) % 200})'
        for robot_id in robot_data.keys()
    }
    
    # Get common timestamps across all robots
    time_stamps = sorted(list(robot_data.values())[0]["timestamps"])
    
    # Calculate bounds
    x_axis_range, z_axis_range = calculate_bounds(robot_data)
    
    # Prepare data for each frame
    frames_data = []
    for frame_idx in range(len(time_stamps)):
        frame_traces = []
        
        # Add pointcloud traces
        for robot_id, data in robot_data.items():
            # Pointcloud data for this frame
            pointclouds = data["pointclouds"][frame_idx]
            pointcloud_x = [p["x"] for p in pointclouds]
            pointcloud_z = [p["z"] for p in pointclouds]
            pointcloud_classes = [p.get("class", "unknown") for p in pointclouds]
            
            # Map classes to colors
            colors = [class_color_map.get(cls, 'rgba(150, 150, 150, 0.7)') for cls in pointcloud_classes]
            
            # Add pointcloud trace
            frame_traces.append({
                "type": "scatter",
                "x": pointcloud_x,
                "y": pointcloud_z,
                "mode": "markers",
                "marker": {
                    "color": colors,
                    "size": 4,
                    "opacity": 0.7
                },
                "name": f"{robot_id} (LiDAR)",
                "showlegend": False,
                "hoverinfo": "none"
            })
        
        # Add robot path traces
        for robot_id, data in robot_data.items():
            robot_color = robot_colors[robot_id]
            
            # Collect all positions up to current frame for path
            path_x = [pos[0] for pos in data["robot_positions"][:frame_idx+1]]
            path_y = [pos[1] for pos in data["robot_positions"][:frame_idx+1]]
            
            frame_traces.append({
                "type": "scatter",
                "x": path_x,
                "y": path_y,
                "mode": "lines",
                "line": {"color": robot_color, "width": 2},
                "name": f"{robot_id} (Path)",
                "showlegend": False,
                "hoverinfo": "none"
            })
        
        # Add robot shape traces
        for robot_id, data in robot_data.items():
            robot_color = robot_colors[robot_id]
            
            # Robot position and orientation for this frame
            robot_pos = data["robot_positions"][frame_idx]
            robot_orient = data["robot_orientations"][frame_idx]
            
            # Create robot shape
            shape_x, shape_y = create_robot_shape(
                robot_pos[0], robot_pos[1], robot_orient
            )
            
            frame_traces.append({
                "type": "scatter",
                "x": shape_x,
                "y": shape_y,
                "mode": "lines",
                "fill": "toself",
                "fillcolor": robot_color,
                "line": {"color": robot_color, "width": 1},
                "name": f"{robot_id}",
                "text": f"{robot_id} @ {time_stamps[frame_idx]:.2f}",
                "hoverinfo": "text"
            })
        
        frames_data.append({
            "name": str(frame_idx),
            "data": frame_traces,
            "timestamp": time_stamps[frame_idx]
        })
    
    # Create initial traces for legend
    init_traces = []
    
    # Add class legend traces
    for class_name, color in class_color_map.items():
        init_traces.append({
            "type": "scatter",
            "x": [None],
            "y": [None],
            "mode": "markers",
            "marker": {
                "size": 10,
                "color": color
            },
            "name": f"{class_name}",
            "showlegend": True
        })
    
    # Add robot legend traces
    for robot_id, color in robot_colors.items():
        init_traces.append({
            "type": "scatter",
            "x": [None],
            "y": [None],
            "mode": "lines",
            "line": {"color": color, "width": 2},
            "name": f"{robot_id}",
            "showlegend": True
        })
    
    return {
        "frames": frames_data,
        "init_traces": init_traces,
        "x_axis_range": x_axis_range,
        "z_axis_range": z_axis_range,
        "time_stamps": time_stamps,
        "robot_colors": robot_colors
    }

def create_html_visualization(robot_data, class_color_map, frame_step, time_step, save_path, filename="AMR LiDAR Visualization.html"):
    """Create a self-contained HTML visualization."""
    # Prepare visualization data
    viz_data = prepare_visualization_data(robot_data, class_color_map)
    
    # Convert data to JSON for JavaScript
    import json
    json_data = json.dumps(viz_data)
    
    # Calculate playback speed options dynamically based on time_step
    # Define desired speed multipliers (relative to real-time)
    speed_multipliers = [0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0]
    
    # Calculate corresponding millisecond delays for each multiplier
    # Formula: delay_ms = (time_step * 1000) / multiplier
    speed_options = []
    for multiplier in speed_multipliers:
        delay_ms = round((time_step * 1000) / multiplier)
        speed_options.append({
            "value": delay_ms,
            "label": f"{multiplier:.2f}x",
            "multiplier": multiplier
        })
    
    # Create the options HTML
    speed_options_html = ""
    for option in speed_options:
        # Make 1.0x the default selected option
        selected = " selected" if abs(option["multiplier"] - 1.0) < 0.01 else ""
        speed_options_html += f'<option value="{option["value"]}"{selected}>{option["label"]}</option>\n'
    
    # Create HTML content
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>AMR Movement with LiDAR Data Visualization</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
            border-radius: 5px;
        }}
        h1 {{
            color: #333;
            text-align: center;
        }}
        .graph-container {{
            height: 800px;
            width: 100%;
            margin-bottom: 20px;
        }}
        .controls {{
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            gap: 20px;
            margin-bottom: 20px;
        }}
        .slider-container {{
            flex: 1;
            min-width: 300px;
        }}
        .button-container {{
            display: flex;
            gap: 10px;
        }}
        button {{
            padding: 8px 16px;
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
        }}
        button:hover {{
            background-color: #45a049;
        }}
        .time-display {{
            font-size: 16px;
            font-weight: bold;
            border: 1px solid #ddd;
            padding: 8px 12px;
            border-radius: 4px;
            background: #f9f9f9;
        }}
        .slider {{
            width: 100%;
        }}
        label {{
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
        }}
        .info-panel {{
            margin-top: 10px;
            padding: 10px;
            background-color: #f0f8ff;
            border-radius: 4px;
            font-size: 14px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1 style = "font-size: 16px; padding-top: 20px">AMR Movement with LiDAR Data Visualization</h1>
        
        <div class="graph-container" id="amr-graph"></div>
        
        <div class="controls">
            <div class="time-display" id="time-display">Time: 0.00s</div>
            
            <div class="slider-container">
                <label for="frame-slider">Timeline:</label>
                <input type="range" id="frame-slider" class="slider" min="0" max="0" value="0">
            </div>
            
            <div class="slider-container">
                <label for="speed-slider">Playback Speed:</label>
                <select id="speed-slider">
                    {speed_options_html}
                </select>
            </div>
            
            <div class="button-container">
                <button id="play-button">Play</button>
                <button id="reset-button">Reset</button>
            </div>
        </div>
        
        <div class="info-panel">
            <p><strong>Frame Step:</strong> {frame_step} | <strong>Time Step:</strong> {time_step:.4f}s</p>
            <p>Playback speeds are calculated relative to real-time, where 1.0x represents the real-time speed.</p>
        </div>
    </div>

    <script>
        // Load data
        const vizData = {json_data};
        
        // Setup variables
        let currentFrame = 0;
        let isPlaying = false;
        let animationId = null;
        let lastTime = 0;
        const timeStep = {time_step};
        let playbackSpeed = {speed_options[3]["value"]}; // Default to 1.0x
        
        // DOM elements
        const graphDiv = document.getElementById('amr-graph');
        const frameSlider = document.getElementById('frame-slider');
        const speedSlider = document.getElementById('speed-slider');
        const playButton = document.getElementById('play-button');
        const resetButton = document.getElementById('reset-button');
        const timeDisplay = document.getElementById('time-display');
        
        // Set up slider
        frameSlider.max = vizData.frames.length - 1;
        
        // Create initial plot
        function createPlot() {{
            const layout = {{
                xaxis: {{
                    title: 'X Position (Unity)',
                    range: vizData.x_axis_range,
                    scaleanchor: 'y',
                    scaleratio: 1,
                    fixedrange: true,
                    constrain: 'domain'
                }},
                yaxis: {{
                    title: 'Z Position (Unity)',
                    range: vizData.z_axis_range,
                    fixedrange: true,
                    constrain: 'domain'
                }},
                legend: {{
                    x: 1.05,
                    y: 1,
                    xanchor: 'left',
                    yanchor: 'top'
                }},
                margin: {{l: 50, r: 50, t: 50, b: 50}},
                plot_bgcolor: 'rgba(240, 240, 240, 0.8)',
                height: 800,
                dragmode: false
            }};
            
            Plotly.newPlot(graphDiv, vizData.init_traces, layout);
            updateFrame(0);
        }}
        
        // Update to a specific frame
        function updateFrame(frameIndex) {{
            // Ensure frame index is within bounds
            frameIndex = Math.max(0, Math.min(frameIndex, vizData.frames.length - 1));
            currentFrame = frameIndex;
            
            // Update slider
            frameSlider.value = frameIndex;
            
            // Update time display
            const timestamp = vizData.frames[frameIndex].timestamp;
            timeDisplay.textContent = `Time: ${{timestamp.toFixed(2)}}s`;
            
            // Update plot with new data
            Plotly.animate(graphDiv, {{
                data: vizData.frames[frameIndex].data,
                traces: Array.from({{length: vizData.frames[frameIndex].data.length}}, (_, i) => i),
                layout: {{}}
            }}, {{
                transition: {{duration: 0}},
                frame: {{duration: 0, redraw: false}}
            }});
        }}
        
        // Animation function
        function animate(timestamp) {{
            if (!lastTime) lastTime = timestamp;
            
            const elapsed = timestamp - lastTime;
            
            if (elapsed > playbackSpeed) {{
                // Update to next frame
                if (currentFrame < vizData.frames.length - 1) {{
                    updateFrame(currentFrame + 1);
                }} else {{
                    // Loop back to beginning
                    updateFrame(0);
                }}
                
                lastTime = timestamp;
            }}
            
            if (isPlaying) {{
                animationId = requestAnimationFrame(animate);
            }}
        }}
        
        // Play/pause toggle
        function togglePlay() {{
            isPlaying = !isPlaying;
            
            if (isPlaying) {{
                playButton.textContent = 'Pause';
                lastTime = 0;
                animationId = requestAnimationFrame(animate);
            }} else {{
                playButton.textContent = 'Play';
                if (animationId) {{
                    cancelAnimationFrame(animationId);
                    animationId = null;
                }}
            }}
        }}
        
        // Reset to beginning
        function resetAnimation() {{
            if (isPlaying) {{
                togglePlay();
            }}
            updateFrame(0);
        }}
        
        // Event listeners
        playButton.addEventListener('click', togglePlay);
        resetButton.addEventListener('click', resetAnimation);
        
        frameSlider.addEventListener('input', () => {{
            updateFrame(parseInt(frameSlider.value, 10));
        }});
        
        speedSlider.addEventListener('change', () => {{
            playbackSpeed = parseFloat(speedSlider.value);
        }});
        
        // Initialize
        createPlot();
    </script>
</body>
</html>
"""
    
    # Save HTML file
    os.makedirs(save_path, exist_ok=True)
    html_path = os.path.join(save_path, filename)
    
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"\n✅ HTML visualization saved to {html_path}")
    print(f"   Time step: {time_step:.4f}s (based on frame_step of {frame_step})")
    print(f"   Playback speed options: {', '.join([opt['label'] for opt in speed_options])}")

if __name__ == "__main__":
    main()