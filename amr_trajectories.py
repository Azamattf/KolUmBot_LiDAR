
# # Animate AMR movement in the x-y plane from JSON step files

# In[1]:


import os
import json
from natsort import natsorted  # Natural sorting
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.widgets import Button
import glob
import itertools
from IPython.display import HTML


# In[2]:


def main():
    # Settings
    solo_nr = 2                 # specify data set

    # define location of all files
    # path_to_dataset = "d:\\Works_Experience\\FML_Florian\\Kolumbot\\Lidar_visualization\\solo_" + str(solo_nr) + "\\"  # on Florian's PC
    path_to_dataset = "d:\\Works_Experience\\FML_Florian\\Kolumbot\\Lidar_visualization\\solo_" + str(solo_nr) + "\\"  # on Azamat's PC
    path_to_images = os.path.join(path_to_dataset, "sequence.0")
    save_path = os.path.join(path_to_dataset, "Export", "AMR animation.mp4")
    
    # Process JSON files and animate robot trajectory

    frame_step = 30

    json_files = load_json_files(path_to_images, frame_step)
    position_data = read_positions(json_files)
    animate_robots(position_data, 0.05, save_path, frame_step)
    


# In[3]:


def load_json_files(path_to_jsons, frame_step):
    data = []
    json_files = [f for f in natsorted(os.listdir(path_to_jsons)) if f.endswith(".json")]
    sampled_files = json_files[::frame_step]  # Sample files based on frame_step

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

    # Check first 5 file names
    print("\nFirst 5 files:")
    print(f"{json_files[:5]} ...")  # Print first 5 positions for each robot

    print(f"\nFirst 5 sampled files (Frame step = {frame_step}):")
    print(f"{sampled_files[:5]} ...")  # Print first 5 positions for each robot

    return data


# In[4]:

def read_positions(data):
    robot_positions = {}
    print("")
    for frame_index, frame in tqdm(list(enumerate(data)), ascii="░▒█", desc="Processing JSON files", unit=" file"):
        timestamp = frame.get("timestamp", 0)
        lidars = [o for o in frame.get("captures", []) if o.get('@type') == 'type.custom/solo.2DLidar' and 'front' in o.get('id')]
        
        for lidar in lidars:
            robot_id = "_".join(lidar["id"].split("_")[:2])  # Extract "AMR_1", "AMR_2"
            position = lidar.get("globalPosition", [])
            
            if position and len(position) == 3:
                x, y = position[0], position[2]  # Use x and z for 2D plotting
            else:
                x, y = None, None  # Mark missing position
            
            if robot_id not in robot_positions:
                robot_positions[robot_id] = []
            
            robot_positions[robot_id].append((timestamp, x, y))

    print("\nExtracted Robot Positions:")
    for robot_id, positions in robot_positions.items():
        print(f"{robot_id}: {positions[:5]} ...")  # Print first 5 positions for each robot

    
    return robot_positions



# In[5]:

def animate_robots(robot_positions, time_interval, save_path=None, frame_step=5):
    # Set up the figure size for the video
    fig, ax = plt.subplots()

    # Set axis limits (same as before)
    all_x = [pos[1] for positions in robot_positions.values() for pos in positions if pos[1] is not None]
    all_y = [pos[2] for positions in robot_positions.values() for pos in positions if pos[2] is not None]

    if not all_x or not all_y:
        print("Error: No valid position data to animate.")
        return

    ax.set_xlim(min(all_x) - (max(all_x) - min(all_x)) * 0.5, max(all_x)+(max(all_x) - min(all_x)) * 0.5)
    ax.set_ylim(min(all_y)*1.1, max(all_y)+(max(all_y) - min(all_y)) * 0.3)
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_title(f"Robot Trajectories (Frame step: {frame_step})")
    plt.gca().set_aspect('equal', adjustable='box')

    # Assign colors and other setup
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    color_map = {robot_id: colors[i % len(colors)] for i, robot_id in enumerate(robot_positions.keys())}

    robot_traces = {robot_id: ax.plot([], [], color=color_map[robot_id], linestyle='-')[0] for robot_id in robot_positions}
    robot_markers = {robot_id: ax.plot([], [], color=color_map[robot_id], marker='o', markersize=6)[0] for robot_id in robot_positions}

    ax.legend(robot_traces.values(), robot_traces.keys(), loc='upper right')

    time_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, fontsize=12, verticalalignment='top')

    timestamps = sorted(set(t for positions in robot_positions.values() for t, _, _ in positions if t is not None))
    if not timestamps:
        print("Error: No timestamps found.")
        return

    max_frames = len(timestamps)
    
    # Remove the progress bar creation from here
    def update(frame):
        """ Update function for animation """
        if frame >= max_frames:
            return []
        current_time = timestamps[frame]
        time_text.set_text(f'Time: {current_time:.1f} s')
        artists = [time_text]
        for robot_id, positions in robot_positions.items():
            past_positions = [(t, x, y) for t, x, y in positions if t <= current_time]
            if past_positions:
                x_data = [pos[1] for pos in past_positions if pos[1] is not None]
                y_data = [pos[2] for pos in past_positions if pos[2] is not None]
                if x_data and y_data:
                    robot_traces[robot_id].set_data(x_data, y_data)
                    robot_markers[robot_id].set_data([x_data[-1]], [y_data[-1]])
                    artists.extend([robot_traces[robot_id], robot_markers[robot_id]])
        # Remove the progress bar update from here
        return artists

    # Create the animation
    anim = FuncAnimation(fig, update, frames=max_frames, interval=time_interval * 1000, blit=False, repeat=False)

    # Save the animation if a save path is provided
    if save_path:
        if not save_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            save_path += ".mp4"
        print(f"\nSaving animation to {save_path}...")
        
        # Create a progress callback
        progress_callback = lambda i, n: progress_bar.update(1)
        
        with tqdm(total=max_frames, desc="Saving Animation", ncols=100, unit="frame") as progress_bar:
            writer = FFMpegWriter(fps=10, bitrate=1800, extra_args=['-vcodec', 'libx264'])
            anim.save(save_path, writer=writer, dpi=200, progress_callback=progress_callback)
        print("✅ Video saved successfully!\n")
    plt.show()


# In[6]:


if __name__ == "__main__":
    main()
