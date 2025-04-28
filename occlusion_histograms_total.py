# coding: utf-8

# # Create _static_ histograms of object classes for each RGB camera from JSON step files

# In[1]:


import os
import json
from natsort import natsorted  # Natural sorting
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict  # provides a default value for missing keys, which eliminates the need 
                                     # to check for key existence before accessing or modifying it


# In[2]:


def main():
    # Settings
    solo_nr = 2                 # specify data set

    # define location of all files
    # path_to_dataset = "d:\\Works_Experience\\FML_Florian\\Kolumbot\\Lidar_visualization\\solo_" + str(solo_nr) + "\\"  # on Florian's PC
    path_to_dataset = "d:\\Works_Experience\\FML_Florian\\Kolumbot\\Lidar_visualization\\solo_" + str(solo_nr) + "\\"  # on Azamat's PC
    path_to_images = path_to_dataset + "sequence.0"
    # path_to_sem_sec_def = path_to_dataset + "semantic_segmentation_definition.json"
    # export_path = path_to_dataset + "Export\\"

    
    
    # Process JSON files and plot histogram
    
    occlusion_data = read_json_files(path_to_images)
    plot_histogram(occlusion_data)
    


# In[3]:


def read_json_files(path_to_images):
    occlusion_data = defaultdict(list)  # Dictionary {sensor_id: [occlusion_values]}

    # Iterate over all JSON files in the directory
    json_files = [f for f in os.listdir(path_to_images) if f.endswith(".json")]
    for filename in tqdm(json_files, desc="Processing JSON files", unit="file"):
        file_path = os.path.join(path_to_images, filename)
        with open(file_path, "r") as f:
            data = json.load(f)
        
        metrics = data.get("metrics", [])
        for metric in metrics:
            sensor_id = metric.get("sensorId", "unknown_sensor") 
            values = metric.get("values", []) 
            for value in values:
                visibility = value.get("visibilityInFrame")
                if visibility is not None:  # excludes the cases where the missing keys are automatically assigned by defaultdict
                    occlusion_data[sensor_id].append(visibility)
    
    return occlusion_data


# In[4]:


def plot_histogram(occlusion_data):
   
    bins = np.arange(0, 1.1, 0.1)  # 10% intervals
    fig, ax = plt.subplots(figsize=(10, 6))

    # Prepare data for stacking
    bottom = np.zeros(len(bins) - 1)
    for sensor_id, occlusion_values in occlusion_data.items():
        hist, _ = np.histogram(occlusion_values, bins=bins)
        ax.bar(bins[:-1]+0.002, hist, width=0.096, align="edge", bottom=bottom, label=sensor_id)
        bottom += hist  # Update the bottom for stacking

    # Set labels and legend
    ax.set_xticks(bins)
    ax.set_xticklabels([f"{int(b * 100)}%" for b in bins], rotation=45, ha="right")
    ax.set_xlabel("Visibility in Frame (Occlusion)")
    ax.set_ylabel("Frequency")
    ax.set_title("Stacked Histogram of Occlusion Values by Sensor")
    ax.legend(title="Camera ID")

    # Show the plot
    plt.tight_layout()
    plt.show()


# In[5]:


if __name__ == "__main__":
    main()

