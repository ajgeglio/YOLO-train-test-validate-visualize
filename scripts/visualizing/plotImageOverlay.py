import argparse
import sys
from pathlib import Path
import pandas as pd
import os
import glob
import random
import numpy as np
# --- 1. SETUP & CONFIGURATION ---
import pathlib
SCRIPT_DIR = pathlib.Path(__file__).parent if '__file__' in locals() else pathlib.Path.cwd()
sys.path.append(str(SCRIPT_DIR.parent.parent / "src"))
try:
    from utils import Utils
    from overlayFunctions import Overlays
except ImportError:
    print(f"WARNING: Could not importfrom {str(SCRIPT_DIR.parent.parent / "src")}. Ensure the path is correct.")

def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Visualize labels and bounding boxes on images.")
    parser.add_argument('--directory', type=str, required=True, help="Path to the parent directory which contains 'images' and 'labels' subdirectories, or 'images.txt' and 'labels.txt' files.")
    parser.add_argument('--list_file', action='store_true', help="If set, read file paths from 'images.txt' and 'labels.txt' in the --directory.")
    # Visualization Mode arguments (Mutually Exclusive Group)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--index', type=int, default=None, help="Plot the image/label at a specific index from the sorted list of files.")
    group.add_argument('--random', action='store_true', help="Plot a random image/label pair.")
    group.add_argument('--all', action='store_true', help="plot and save all the images in the directory")
    # Optional arguments for file path discovery
        # Optional arguments for file type selection
    parser.add_argument('--labels_only', action='store_true', help="Process only label files. Requires --image_width and --image_height.")
    parser.add_argument('--images_only', action='store_true', help="Process only image files.")
    # Arguments required for 'labels_only' mode
    parser.add_argument('--image_width', type=int, default=None, help="Required image width for calculating absolute bounding box coordinates in --labels_only mode.")
    parser.add_argument('--image_height', type=int, default=None, help="Required image height for calculating absolute bounding box coordinates in --labels_only mode.")
    args = parser.parse_args()
    
    # Enforce dimension arguments in labels_only mode
    if args.labels_only and (args.image_width is None or args.image_height is None):
        parser.error("Error: --image_width and --image_height are required in --labels_only mode.")
    return args


def return_path_list(path):
    """Retrieve file paths based on provided list file."""
    try:
        print(f"Using list file: {path}")
        with open(path, 'r', encoding='utf-8-sig') as f:
            path_list = f.read().splitlines()
        assert len(path_list) > 0, "No paths found in the list file."
    except FileNotFoundError:
        raise ValueError(f"List file not found: {path}")
    return sorted(path_list)


def get_file_paths(args):
    """Retrieves file paths based on command-line arguments."""
    dir_path = os.path.abspath(args.directory)
    image_paths, label_paths = [], []
    
    # Get image paths
    if not args.labels_only:
        if args.list_file:
            image_csv = os.path.join(dir_path, "images.txt")
            image_paths = return_path_list(image_csv)
        else:
            image_dir = os.path.join(dir_path, "images")
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff']:
                image_paths.extend(glob.glob(os.path.join(image_dir, ext)))
    
    # Get label paths
    if not args.images_only:
        if args.list_file:
            label_csv = os.path.join(dir_path, "labels.txt")
            label_paths = return_path_list(label_csv)
        else:
            label_dir = os.path.join(dir_path, "labels")
            label_paths.extend(glob.glob(os.path.join(label_dir, '*.txt')))
            
    assert len(image_paths) > 0 or len(label_paths) > 0, \
        f"No image or label files found in the specified directory: {dir_path}"
    assert len(image_paths) == len(label_paths) or args.labels_only or args.images_only, \
        "Mismatch between image and label counts. Ensure both directories have matching files or use --images_only or --labels_only."
    
    print(f"Found {len(image_paths)} image files and {len(label_paths)} label files.")
        
    return sorted(image_paths), sorted(label_paths)


def main():
    """Main function to parse arguments, get paths, and display overlays."""
    args = parse_arguments()
    image_paths, label_paths = get_file_paths(args)
    path_length = max(len(image_paths), len(label_paths))

    # default to random if no user input
    plot_index = [random.randint(0, path_length - 1)]
    
    # 1. Determine the index to plot
    if args.index is not None:
        save_dir = None
        plot_index = [args.index]
        max_index = len(image_paths) - 1 if image_paths else len(label_paths) - 1
        if plot_index[0] < 0 or plot_index[0] > max_index:
            sys.exit(f"Error: Index {plot_index} is out of bounds (0 to {max_index}).")
        print(f"Plotting file at specified index: **{plot_index}**")
        
    elif args.random:
        save_dir = None
        matched_paths = image_paths if image_paths else label_paths
        if not matched_paths:
            sys.exit("Error: Cannot select a random index, no files found.")
            
        plot_index = [random.randint(0, len(matched_paths) - 1)]
        print(f"Plotting a random file. Selected index: **{plot_index}**")

    elif args.all:
        save_dir = os.path.join(os.path.dirname(image_paths[0]), "overlays")
        plot_index = range(len(image_paths))
        print("plotting all images in directory and saving to", save_dir)
        
    else:
        print(f"No --index or --random specified. Plotting the first file (index: 0).")

    # 2. Get the specific paths and plot
    for i in plot_index:
        print(f"\nDisplaying image with overlay...")
        print(f"  Index: {i}")
        if not args.labels_only:
            image_path = image_paths[i]
            print(f"  Image: {os.path.basename(image_path)}")
        elif args.labels_only:
            label_path = label_paths[i]
            label_path = Path(label_path)
            image_path = label_paths[i]
            # lbl = Utils.read_YOLO_lbl(label_path)
            print(f"  Overlaying blank image")
            # print(lbl)
        if not args.images_only:
            label_path = label_paths[i]
            # lbl = Utils.read_YOLO_lbl(label_path)
            lbl = Utils.read_YOLO_lbl(label_path)
            print(f"  Label: {os.path.basename(label_path)}")
            print(lbl)
        elif args.images_only:
            label_path = image_paths[i]
            print(f"  No label selected")

        
        
        plot_img = Overlays.disp_lbl_bbox(image_path, label_path, label_only=args.labels_only, imw=args.image_width, imh=args.image_height, save_path=save_dir)
        if not save_dir:
            plot_img.show()
    
    
if __name__ == "__main__":
    main()