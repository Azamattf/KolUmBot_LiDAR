
# # Animate AMR movement with yaw direction in the x-y plane from JSON step files

# In[1]:

import os
import json
import csv
import plotly.graph_objects as go
import pandas as pd
from natsort import natsorted
from tqdm import tqdm
import numpy as np

# In[2]:


def main():
    # Settings
    solo_nr = 2                 # specify data setunity 

    # define location of all files
    # path_to_dataset = "d:\\Works_Experience\\FML_Florian\\Kolumbot\\Lidar_visualization\\solo_" + str(solo_nr) + "\\"  # on Florian's PC
    path_to_dataset = "d:\\Works_Experience\\FML_Florian\\Kolumbot\\Lidar_visualization\\solo_" + str(solo_nr) + "\\"  # on Azamat's PC
    path_to_images = os.path.join(path_to_dataset, "sequence.0")
    save_path = os.path.join(path_to_dataset, "Export")
    
    # Process JSON files and animate robot trajectory

    frame_step = 40

    json_files = load_json_files(path_to_images, frame_step)
    position_data = read_positions(json_files, save_path)

    animation = AMRAnimation(position_data)
    animation.create_animation(save_path)
    

# %% AMR Class

class AMR:
    def __init__(self, name, trajectory, color):
        self.name = name
        self.trajectory = trajectory
        self.color = color
        self.x_values = []
        self.y_values = []
        
    def get_trace(self):
        # Create an initial trace for the AMR's path
        return go.Scatter(x=[], y=[], mode="lines+markers", name=self.name,
                         line=dict(color=self.color), marker=dict(color=self.color))
    
    def get_position_at_frame(self, frame_idx):
        # Get position and yaw at a specific frame
        if frame_idx < len(self.trajectory):
            x, y = self.trajectory[frame_idx][1], self.trajectory[frame_idx][2]
            yaw_x, yaw_y = self.trajectory[frame_idx][3], self.trajectory[frame_idx][4]
            timestamp = self.trajectory[frame_idx][0]  # Get timestamp from trajectory
            return x, y, yaw_x, yaw_y, timestamp
        elif len(self.trajectory) > 0:
            # Return last known position if beyond trajectory
            x, y = self.trajectory[-1][1], self.trajectory[-1][2]
            yaw_x, yaw_y = self.trajectory[-1][3], self.trajectory[-1][4]
            timestamp = self.trajectory[-1][0]  # Get last timestamp
            return x, y, yaw_x, yaw_y, timestamp
        else:
            # Fallback if trajectory is empty
            return 0, 0, 0, 0, 0
        
    def get_frame(self, frame_idx):
        # Generate a frame showing the AMR's position up to this frame
        # Update position history up to this frame
        self.x_values = []
        self.y_values = []
        
        for i in range(min(frame_idx + 1, len(self.trajectory))):
            x, y = self.trajectory[i][1], self.trajectory[i][2]
            self.x_values.append(x)
            self.y_values.append(y)
            
        # Create a trace for the path (circular dots)
        path_trace = go.Scatter(
            x=self.x_values, y=self.y_values, mode="lines+markers",
            line=dict(color=self.color), marker=dict(color=self.color, size=5),
            name=f"{self.name} Path"
        )
        
        # Create a trace for the robot's current position (square)
        current_x, current_y, _, _, _ = self.get_position_at_frame(frame_idx)
        robot_trace = go.Scatter(
            x=[current_x], y=[current_y], mode="markers",
            marker=dict(color=self.color, symbol="square", size=10),
            name=f"{self.name} Robot"
        )
        
        return [path_trace, robot_trace]

class AMRAnimation:
    def __init__(self, robot_positions, fps=20, frame_step=1):
        self.robot_positions = robot_positions
        self.fps = fps
        self.frame_step = frame_step
        self.amrs = self.create_amrs()
        self.x_range, self.y_range = self.compute_limits()
        self.max_frames = max(len(traj) for traj in self.robot_positions.values())
        
    def create_amrs(self):
        color_palette = ["blue", "red", "green", "purple", "orange", "brown", "cyan", "magenta"]
        return [AMR(name, data, color_palette[i % len(color_palette)]) 
                for i, (name, data) in enumerate(self.robot_positions.items())]
    
    def compute_limits(self):
        x_min = min(min(data[i][1] for i in range(len(data))) for data in self.robot_positions.values())
        x_max = max(max(data[i][1] for i in range(len(data))) for data in self.robot_positions.values())
        y_min = min(min(data[i][2] for i in range(len(data))) for data in self.robot_positions.values())
        y_max = max(max(data[i][2] for i in range(len(data))) for data in self.robot_positions.values())
        return (x_min - 5, x_max + 5), (y_min - 5, y_max + 5)
    
    def get_timestamp_at_frame(self, frame_idx):
        if self.amrs:
            _, _, _, _, timestamp = self.amrs[0].get_position_at_frame(frame_idx)
            return timestamp
        return 0
    
    def format_timestamp(self, timestamp):
        minutes = int(timestamp // 60)
        seconds = int(timestamp % 60)
        return f"{minutes:02d}:{seconds:02d}"
    
    def create_arrow_traces(self, frame_idx):
        # Create arrow traces for all AMRs at a specific frame
        traces = []
        
        for amr in self.amrs:
            x, y, yaw_x, yaw_y, _ = amr.get_position_at_frame(frame_idx)
            
            # Define arrow length and calculate endpoint
            arrow_length = 3
            arrow_x = x + yaw_x * arrow_length
            arrow_y = y + yaw_y * arrow_length
            
            # Create line trace for arrow shaft
            shaft_trace = go.Scatter(
                x=[x, arrow_x],
                y=[y, arrow_y],
                mode='lines',
                line=dict(color=amr.color, width=2),
                showlegend=False,
                hoverinfo='skip'
            )
            
            # Create triangle for arrow head
            head_size = 2
            dx = arrow_x - x
            dy = arrow_y - y
            angle = np.arctan2(dy, dx)
            
            # Calculate arrow head points
            arrow_head_1_x = arrow_x - head_size * np.cos(angle - np.pi/6)
            arrow_head_1_y = arrow_y - head_size * np.sin(angle - np.pi/6)
            arrow_head_2_x = arrow_x - head_size * np.cos(angle + np.pi/6)
            arrow_head_2_y = arrow_y - head_size * np.sin(angle + np.pi/6)
            
            head_trace = go.Scatter(
                x=[arrow_head_1_x, arrow_x, arrow_head_2_x],
                y=[arrow_head_1_y, arrow_y, arrow_head_2_y],
                mode='lines',
                fill='toself',
                fillcolor=amr.color,
                line=dict(color=amr.color, width=1),
                showlegend=False,
                hoverinfo='skip'
            )
            
            traces.extend([shaft_trace, head_trace])
            
        return traces
    
    def create_animation(self, save_path):
        # Get initial position traces
        initial_traces = []
        for amr in self.amrs:
            initial_traces.extend(amr.get_frame(0))  # AMR position traces
        
        # Add initial arrow traces
        initial_traces.extend(self.create_arrow_traces(0))
        
        frames = []
        slider_steps = []
        
        for frame_idx in range(0, self.max_frames, self.frame_step):
            # Get updated traces for AMR positions
            frame_data = []
            for amr in self.amrs:
                frame_data.extend(amr.get_frame(frame_idx))
            
            # Add arrow traces for this frame
            frame_data.extend(self.create_arrow_traces(frame_idx))
            
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
                title="AMR Animation with Yaw",
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
                }]
            ),
            frames=frames
        )
        
        fig.show()
        
        # Save animation as HTML
        os.makedirs(save_path, exist_ok=True)
        filepath = os.path.join(save_path, "AMR_Animation_oop.html")
        fig.write_html(filepath)


# In[3]:


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


# %% Unity to ROS to yaw conversion

def unity_quaternion_to_ros_yaw(q_unity):
    """ Convert Unity quaternion (x, y, z, w) to ROS quaternion (x, z, y, w) """
    x, y, z, w = q_unity
    q_ros = np.array([x, z, y, w])

    """ Convert ROS quaternion to a 2D heading direction (yaw) in the x-y plane. """
    heading = np.arctan2(2 * (q_ros[3] * q_ros[2] + q_ros[0] * q_ros[1]), 1 - 2 * ( q_ros[1]**2 +  q_ros[2]**2))
    dy = np.cos(heading)  # simply dx and dy are interchanged for correct plotting (?????)
    dx = np.sin(heading)
    return dx, dy


# %% Extract positions and yaw 

def read_positions(data, savepath):
    robot_positions = {}
    print("")
    for frame_index, frame in tqdm(list(enumerate(data)), ascii="░▒█", desc="Processing JSON files", unit=" file"):
        timestamp = frame.get("timestamp", 0)
        lidars = [o for o in frame.get("captures", []) if o.get('@type') == 'type.custom/solo.2DLidar' and 'front' in o.get('id')]
        
        for lidar in lidars:
            robot_id = "_".join(lidar["id"].split("_")[:2])  # Extract "AMR_1", "AMR_2"
            position = lidar.get("globalPosition", [])
            quaternion = lidar.get("globalRotation", [])
            
            if position and len(position) == 3 and quaternion and len(quaternion) == 4:
                x, y, = position[0], position[2]  # Use x and z for 2D plotting
                dx, dy = unity_quaternion_to_ros_yaw(quaternion)

            else:
                x, y, dx, dy = None, None, None, None  # Mark missing data
            

            if robot_id not in robot_positions:
                robot_positions[robot_id] = []
            
            robot_positions[robot_id].append((timestamp, x, y, dx, dy))

    print("\nExtracted Robot Positions and Headings:")
    for robot_id, positions in robot_positions.items():
        print(f"{robot_id}: {positions[:5]} ...")  # Print first 5 positions for each robot

    # Save data as CSV

    def save_robot_positions_to_csv(robot_positions, filename="AMR_positions.csv"):
    
        # Create the directory if it doesn't exist
        os.makedirs(savepath, exist_ok=True)
        filepath = os.path.join(savepath, filename)
        
        # Write data to CSV
        with open(filepath, mode="w", newline="") as file:
            writer = csv.writer(file)
            
            # Write header
            writer.writerow(["Robot_ID", "Timestamp", "X", "Y", "Yaw_X", "Yaw_Y"])
            
            # Write data
            for robot_id, positions in robot_positions.items():
                for position in positions:
                    writer.writerow([robot_id] + list(position))  # Unpack tuple into CSV row

        print(f"\n✅ Robot positions saved to {filepath}")

    
    save_robot_positions_to_csv(robot_positions)

   
    return robot_positions




# In[6]:


if __name__ == "__main__":
    main()
