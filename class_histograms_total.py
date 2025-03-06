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
    
    label_data = read_json_files(path_to_images)
    plot_histogram(label_data)
    


# In[3]:


def read_json_files(path_to_images):
    label_data = defaultdict(lambda: defaultdict(int))  # {camera_id: {label_name: count}}

    # Iterate over all JSON files in the directory
    json_files = [f for f in os.listdir(path_to_images) if f.endswith(".json")]
    for filename in tqdm(json_files, desc="Processing JSON files", unit="file"):
        file_path = os.path.join(path_to_images, filename)
        with open(file_path, "r") as f:
            data = json.load(f)
        
        step = data.get("step", None)
        timestemp = data.get("timestamp", None)   # in seconds
        captures = data.get("captures", [])
        read_cameras = list(o for o in captures if o['@type'] == 'type.unity.com/unity.solo.RGBCamera')
        
        for camera in read_cameras:
            camera_id = camera.get("id", "unknown_camera")
            annotations = camera.get("annotations", [])
        
            for annotation in annotations:
                if annotation.get('@type') == 'type.unity.com/unity.solo.BoundingBox2DAnnotation':
                    values = annotation.get("values", [])
                    for value in values:
                        label_name = value.get("labelName")
                        if label_name:   # ensure the labelName exists to avoid counting automatically initiated instances by defaultdict 
                            label_data[camera_id][label_name] += 1
            
    return label_data


# In[4]:


def plot_histogram(label_data):
    
    fig, ax = plt.subplots(figsize=(10, 6))

    # Extract unique labels across all cameras
    all_labels = set()
    for camera_id in label_data:
        all_labels.update(label_data[camera_id].keys())
    all_labels = sorted(all_labels)  # Sort labels for consistency
    x = np.arange(len(all_labels))  # X-axis positions for labels

    # Prepare data for stacking
    bottom = np.zeros(len(all_labels))
    for camera_id, label_counts in label_data.items():
        counts = np.array([label_counts.get(label, 0) for label in all_labels])
        ax.bar(x, counts, bottom=bottom, label=camera_id)
        bottom += counts  # Update the bottom for stacking

    # Set x-axis labels and ticks
    ax.set_xticks(x)
    ax.set_xticklabels(all_labels, rotation=45, ha="right")
    ax.set_xlabel("Labels")
    ax.set_ylabel("Counts")
    ax.set_title("Stacked Label Distribution by Camera")

    # Add legend
    ax.legend(title="Camera ID")

    # Show the plot
    plt.tight_layout()
    plt.show()


# In[5]:


if __name__ == "__main__":
    main()

