# Histograms Generator
There are several histogram generators in the project repo, 2 of which are of importance:
- `class_histograms_total.py`
- `occlusion_histograms_total.py`
This document describes only `class_histograms_total.py` because the latter does the same focusing on occlusion data, instead of the class data.

# `class_histograms_total.py`
This Python script analyzes JSON step files from Unity Solo simulations to create static histograms of object classes detected by RGB cameras. It visualizes the distribution of different object classes across all cameras in the simulation.

## Features
- Processes JSON files from RGB camera captures
- Counts bounding box annotations per object class for each camera
- Creates a **stacked histogram** visualization showing class distribution across cameras
- Adds a legend for AMR IDs

## Configuration
Edit these parameters in `main()`:
- `solo_nr`: Specify which dataset to use
- `path_to_dataset`: Path to the dataset folder
- `path_to_images`: Path to JSON sequence files

## Key Functions
- `read_json_files()`: Processes JSON files and counts object classes by camera
- `plot_histogram()`: Creates a stacked bar chart visualization of the class distribution

## Visualization Features
- Stacked bars showing counts of each object class
- Color-coded by camera ID
- Rotated labels for better readability
- Automatic legend generation

## Output
- Interactive matplotlib visualization showing the distribution of object classes
- Classes are sorted alphabetically on the x-axis
- Y-axis shows the total count of object instances
