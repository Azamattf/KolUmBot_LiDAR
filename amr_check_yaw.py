
# # Check AMR yaw    data  in the x-y plane from JSON step files

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
    # Data path

    path_to_dataset = "d:\\Works_Experience\\FML_Florian\\Kolumbot\\Lidar_visualization\\Unity_v3s\\"
    path_to_images = os.path.join(path_to_dataset, "sequence.0_json_only")
    save_path = os.path.join(path_to_dataset, "Export")
    path_to_sem_def = os.path.join(path_to_dataset, "class_definition_semantic_segmentation.json")
      
    # Process JSON files

    frame_step = 1

    json_files = load_json_files(path_to_images, frame_step)
    position_data = read_positions(json_files, save_path)


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

    """ Convert ROS quaternion to a 2D yaw in the x-y plane. """
    heading = np.arctan2(2 * (q_ros[3] * q_ros[2] + q_ros[0] * q_ros[1]), 1 - 2 * ( q_ros[1]**2 +  q_ros[2]**2))
    dy = np.cos(heading)
    dx = np.sin(heading)

    yaw = np.arctan2(dx,dy)

    return yaw


# %% Extract positions and yaw 

def read_positions(data, savepath):
    robot_positions = {}
    print("")
    for frame_index, frame in tqdm(list(enumerate(data)), ascii="░▒█", desc="Processing JSON files", unit=" file"):
        timestamp = frame.get("timestamp", 0)
        lidars = [o for o in frame.get("captures", []) if o.get('@type') == 'type.custom/solo.2DLidar' and 'front' in o.get('id')]
        
        for lidar in lidars:
            robot_id = "_".join(lidar["id"].split("_")[:2])  # Extract "AMR_1", "AMR_2"
            # position = lidar.get("robotPosition", [])
            quaternion = lidar.get("robotRotation", [])
            yaw_given = lidar.get("robotRotationEuler")
            
            # x, y, = position[0], position[2]  # Use x and z for 2D plotting
            yaw_from_quat = unity_quaternion_to_ros_yaw(quaternion)

            

            if robot_id not in robot_positions:
                robot_positions[robot_id] = []
            
            robot_positions[robot_id].append((timestamp, yaw_given, yaw_from_quat))

    print("\nExtracted Data:")
    for robot_id, positions in robot_positions.items():
        print(f"{robot_id}: {positions[:5]} ...")  # Print first 5 positions for each robot

    # Save data as CSV

    def save_robot_positions_to_csv(robot_positions, filename="AMR_Check_Yaw.csv"):
    
        # Create the directory if it doesn't exist
        os.makedirs(savepath, exist_ok=True)
        filepath = os.path.join(savepath, filename)
        
        # Write data to CSV
        with open(filepath, mode="w", newline="") as file:
            writer = csv.writer(file)
            
            # Write header
            writer.writerow(["Robot_ID", "Timestamp", "Yaw_Given", "Yaw_Calculated"])
            
            # Write data
            for robot_id, positions in robot_positions.items():
                for position in positions:
                    writer.writerow([robot_id] + list(position))  # Unpack tuple into CSV row

        print(f"\n✅ Robot yaw data saved to {filepath}")

    
    save_robot_positions_to_csv(robot_positions)
   
    return robot_positions




# In[6]:


if __name__ == "__main__":
    main()
