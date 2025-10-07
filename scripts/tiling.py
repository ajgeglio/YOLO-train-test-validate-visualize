import glob
import os
import cv2
import numpy as np
import argparse
import tqdm

def parse_arguments():
    """Parses command-line arguments for the tiling script."""
    parser = argparse.ArgumentParser(description="Tiling datasets for training")
    parser.add_argument('--directory', required=True, help="Directory containing images and/or labels folders.")
    parser.add_argument('--list_csv', action="store_true", help='Use images.csv and labels.csv for file paths.')
    parser.add_argument('--image_width', type=int, default=None, help="Image width (required in labels_only mode).")
    parser.add_argument('--image_height', type=int, default=None, help="Image height (required in labels_only mode).")
    parser.add_argument('--tile_size_x', type=int, default=1672)
    parser.add_argument('--tile_size_y', type=int, default=1307)
    parser.add_argument('--overlap_x', type=int, default=460)
    parser.add_argument('--overlap_y', type=int, default=461)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--images_only', action="store_true", help='Only process images.')
    group.add_argument('--labels_only', action="store_true", help="Only process labels.")
    return parser.parse_args()

def get_file_paths(args):
    """Retrieves file paths based on command-line arguments."""
    dir = os.path.abspath(args.directory)
    image_paths, label_paths = [], []
    
    # Get image paths
    if not args.labels_only:
        if args.list_csv:
            image_csv = os.path.join(dir, "images.csv")
            if os.path.exists(image_csv):
                with open(image_csv, 'r') as f:
                    image_paths = f.read().splitlines()
        else:
            image_dir = os.path.join(dir, "images")
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff']:
                image_paths.extend(glob.glob(os.path.join(image_dir, ext)))
    
    # Get label paths
    if not args.images_only:
        if args.list_csv:
            label_csv = os.path.join(dir, "labels.csv")
            if os.path.exists(label_csv):
                with open(label_csv, 'r') as f:
                    label_paths = f.read().splitlines()
        else:
            label_dir = os.path.join(dir, "labels")
            label_paths.extend(glob.glob(os.path.join(label_dir, '*.txt')))

    return sorted(image_paths), sorted(label_paths)

def main():
    args = parse_arguments()
    
    # Enforce dimension arguments in labels_only mode
    if args.labels_only and (args.image_width is None or args.image_height is None):
        raise ValueError("Error: --image_width and --image_height are required in --labels_only mode.")

    image_paths, label_paths = get_file_paths(args)

    if not image_paths and not label_paths:
        print("Error: No image or label files found. Exiting.")
        return
    
    # Check for consistency when both are expected
    if not args.images_only and not args.labels_only and len(image_paths) != len(label_paths):
        raise ValueError("Error: Mismatch between image and label counts. Exiting.")

    base_dir = os.path.join(os.path.abspath(os.path.dirname(args.directory)), "tiled")
    tiled_image_path = os.path.join(base_dir, 'images')
    tiled_label_path = os.path.join(base_dir, 'labels')

    if not args.labels_only:
        os.makedirs(tiled_image_path, exist_ok=True)
    if not args.images_only:
        os.makedirs(tiled_label_path, exist_ok=True)
        
    # Prepare the iterator for the main loop
    if args.labels_only:
        paths_to_process = label_paths
    else:
        paths_to_process = image_paths

    for file_path in tqdm.tqdm(paths_to_process):
        img_path, lbl_path = None, None
        
        # Determine paths based on mode
        if args.labels_only:
            lbl_path = file_path
            base_name = os.path.splitext(os.path.basename(lbl_path))[0]
            img_w, img_h = args.image_width, args.image_height
            img = None # No image to read
        else:
            img_path = file_path
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Could not read image at {img_path}. Skipping.")
                continue
            img_h, img_w, _ = img.shape
            
            # Find the corresponding label path
            if not args.images_only:
                # Assumes 'labels' folder is directly under the argument's directory
                labels_dir = os.path.join(os.path.abspath(args.directory), 'labels')
                lbl_path = os.path.join(labels_dir, base_name + '.txt')

        # Dynamic tile sizing based on detected or provided image height
        tile_size_x, tile_size_y, overlap_x, overlap_y = args.tile_size_x, args.tile_size_y, args.overlap_x, args.overlap_y
        
        if img_h == 3000:
            tile_size_x, tile_size_y, overlap_x, overlap_y = 1672, 1307, 460, 460
        elif img_h == 2176:
            tile_size_x, tile_size_y, overlap_x, overlap_y = 1672, 1307, 460, 438
        
        labels_data = []
        if lbl_path and os.path.exists(lbl_path):
            with open(lbl_path, 'r') as f:
                labels_data = [line.strip().split() for line in f.readlines()]

        y = 0
        while True:
            y_start = y
            y_end = y + tile_size_y
            if y_end > img_h:
                y_start = img_h - tile_size_y
                y_end = img_h

            x = 0
            while True:
                x_start = x
                x_end = x + tile_size_x
                if x_end > img_w:
                    x_start = img_w - tile_size_x
                    x_end = img_w

                if not args.labels_only:
                    tile = img[y_start:y_end, x_start:x_end]
                    tile_name = f"{base_name}_{x_start}_{y_start}.png"
                    cv2.imwrite(os.path.join(tiled_image_path, tile_name), tile)

                if not args.images_only:
                    new_labels = []
                    
                    # Tile dimensions for normalization and clipping
                    tile_width_px = x_end - x_start
                    tile_height_px = y_end - y_start
                    
                    for label in labels_data:
                        class_id = int(label[0])
                        # Convert YOLO normalized coordinates to absolute pixel coordinates
                        x_center, y_center, width, height = [float(label[i]) * dim for i, dim in enumerate([img_w, img_h, img_w, img_h], 1)]
                        
                        # Calculate original object boundary coordinates
                        obj_x_min = x_center - width / 2
                        obj_y_min = y_center - height / 2
                        obj_x_max = x_center + width / 2
                        obj_y_max = y_center + height / 2
                        
                        # --- 1. Intersection Check (Relaxed Filter) ---
                        # Check if the object intersects the tile at all. 
                        # Only exclude if the object is entirely outside the tile boundaries.
                        is_intersecting = (
                            obj_x_min < x_end and obj_x_max > x_start and
                            obj_y_min < y_end and obj_y_max > y_start
                        )

                        if is_intersecting:
                            
                            # --- 2. Clip Coordinates at Tile Boundary ---
                            
                            # Calculate the intersection coordinates in pixel space
                            clipped_x_min = max(obj_x_min, x_start)
                            clipped_y_min = max(obj_y_min, y_start)
                            clipped_x_max = min(obj_x_max, x_end)
                            clipped_y_max = min(obj_y_max, y_end)

                            # Calculate new width and height of the clipped box
                            clipped_width = clipped_x_max - clipped_x_min
                            clipped_height = clipped_y_max - clipped_y_min

                            # Calculate new center point of the clipped box
                            clipped_x_center = clipped_x_min + clipped_width / 2
                            clipped_y_center = clipped_y_min + clipped_height / 2

                            # --- 3. Normalize to Tile Dimensions (0.0 to 1.0) ---
                            
                            # Normalize the new clipped box coordinates relative to the tile size
                            new_x_center = (clipped_x_center - x_start) / tile_width_px
                            new_y_center = (clipped_y_center - y_start) / tile_height_px
                            new_width = clipped_width / tile_width_px
                            new_height = clipped_height / tile_height_px

                            # Final sanity check: ensure no box is too small after clipping (e.g., just a single pixel)
                            # You might want to skip boxes smaller than a threshold, but for now, we include all.
                            
                            new_labels.append(f"{class_id} {new_x_center} {new_y_center} {new_width} {new_height}")
                    
                    tile_label_name = f"{base_name}_{x_start}_{y_start}.txt"
                    with open(os.path.join(tiled_label_path, tile_label_name), 'w') as f:
                        f.write('\n'.join(new_labels))

                x += tile_size_x - overlap_x
                if x_start == img_w - tile_size_x:
                    break
            
            y += tile_size_y - overlap_y
            if y_start == img_h - tile_size_y:
                break

if __name__ == '__main__':
    main()