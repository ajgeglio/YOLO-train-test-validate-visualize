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
    parser.add_argument('--list_file', action="store_true", help='Use images.txt and labels.txt for file paths.')
    parser.add_argument('--image_width', type=int, default=None, help="Image width (required in labels_only mode).")
    parser.add_argument('--image_height', type=int, default=None, help="Image height (required in labels_only mode).")
    parser.add_argument('--tile_size_x', type=int, default=1672)
    parser.add_argument('--tile_size_y', type=int, default=1307)
    parser.add_argument('--overlap_x', type=int, default=460)
    parser.add_argument('--overlap_y', type=int, default=461)
    parser.add_argument('--resume', action="store_true", help="resume tiling if interrupted")
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--images_only', action="store_true", help='Only process images.')
    group.add_argument('--labels_only', action="store_true", help="Only process labels.")
    return parser.parse_args()

def return_path_list(path):
    """Retrieve test images based on provided arguments."""
    try:
        print("Using list file", path)
        # --- Change made here ---
        # Use 'utf-8-sig' to automatically handle and remove the Byte Order Mark (BOM)
        with open(path, 'r', encoding='utf-8-sig') as f:
            path_list = f.read().splitlines()
        # ------------------------
        assert len(path_list) > 0, "No image paths found in csv"
    except FileNotFoundError:
        raise ValueError("Must provide an image directory or a path to a csv listing filepaths of images")
    return sorted(path_list)

def get_file_paths(args):
    """Retrieves file paths based on command-line arguments."""
    dir = os.path.abspath(args.directory)
    image_paths, label_paths = [], []
    
    # Get image paths
    if not args.labels_only:
        if args.list_file:
            image_csv = os.path.join(dir, "images.txt")
            image_paths = return_path_list(image_csv)
        else:
            image_dir = os.path.join(dir, "images")
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff']:
                image_paths.extend(glob.glob(os.path.join(image_dir, ext)))
    
    # Get label paths
    if not args.images_only:
        if args.list_file:
            label_csv = os.path.join(dir, "labels.txt")
            label_paths = return_path_list(label_csv)
        else:
            label_dir = os.path.join(dir, "labels")
            label_paths.extend(glob.glob(os.path.join(label_dir, '*.txt')))
    assert len(image_paths) > 0 or len(label_paths) > 0, "No image or label files found in the specified directory."
    assert len(image_paths) == len(label_paths) or args.labels_only or args.images_only, \
        "Mismatch between image and label counts. Ensure both directories have matching files or use --images_only or --labels_only."
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

    if args.resume:
        # 1. Determine which original files have been completed
        if not args.labels_only:
            # We must use the tiled image directory to check for completion
            tiles_complete_w_ext = os.listdir(tiled_label_path)
        elif not args.images_only:
            # If in labels_only mode, we must use the tiled label directory
            tiles_complete_w_ext = os.listdir(tiled_label_path)
        else:
            # Should not happen if not args.labels_only is checked above, but safe to exclude
            print("Cannot resume in images_only mode without prior label processing.")
            return

        # Rsplit works correctly for 'basename_x_y.ext'
        # basenames_complete will hold the original full basename: 'PI_1718720450_372_Iver3069'
        basenames_complete_tiled = list(map(lambda x: x.rsplit('_', 2)[0], tiles_complete_w_ext))
        
        # 2. Extract unique completed basenames
        # The set() operation removes duplicates (from multiple tiles per image)
        unique_completed_basenames = set(basenames_complete_tiled)
        
        # 3. Remove the last basename, as it is likely only partially tiled (the source of the interruption)
        # Find the max basename to ensure we remove the one that was actively processing
        # Note: This assumes the tiled filenames are still processing in alphabetical order.
        if unique_completed_basenames:
            last_processed_basename = max(unique_completed_basenames)
            unique_completed_basenames.remove(last_processed_basename)

        # 4. Determine Remaining Original Paths
        
        # Get the set of basenames from ALL original paths
        original_image_basenames = {os.path.splitext(os.path.basename(p))[0]: p for p in image_paths}
        original_label_basenames = {os.path.splitext(os.path.basename(p))[0]: p for p in label_paths}

        # Calculate the basenames that still need tiling
        remaining_basenames = set(original_image_basenames.keys()).difference(unique_completed_basenames)

        # Rebuild image_paths and label_paths by filtering the original lists.
        # This preserves the original file extensions and full file paths (crucial for --list_file)
        
        # For images:
        if image_paths:
            # Filter the original image_paths list to keep only those that are in remaining_basenames
            image_paths = [original_image_basenames[bn] for bn in remaining_basenames if bn in original_image_basenames]

        # For labels:
        if label_paths:
            # Filter the original label_paths list to keep only those that are in remaining_basenames
            label_paths = [original_label_basenames[bn] for bn in remaining_basenames if bn in original_label_basenames]
            
        # Re-sort to maintain order for paired processing
        image_paths.sort()
        label_paths.sort()
        
        print(f"Resuming: {len(image_paths)} images and {len(label_paths)} labels remaining to process.")
        
    # Prepare the iterator for the main loop
    if args.labels_only:
        paths_to_process = label_paths
    else:
        paths_to_process = image_paths

    # Dynamic tile sizing based on detected or provided image height
    # Initialize to defaults (or args, which use defaults if not provided)
    tile_size_x, tile_size_y, overlap_x, overlap_y = args.tile_size_x, args.tile_size_y, args.overlap_x, args.overlap_y
    
    tiling_height_check = args.image_height if args.labels_only else None
    
    if tiling_height_check == 3000:
        # Override defaults for 4096x3000 (3x3 grid)
        tile_size_x, tile_size_y, overlap_x, overlap_y = 1672, 1307, 460, 460
    elif tiling_height_check == 2176:
        # Override defaults for 4096x2176 (3x2 grid)
        tile_size_x, tile_size_y, overlap_x, overlap_y = 1672, 1307, 460, 438

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
            if img_h == 3000:
                tile_size_x, tile_size_y, overlap_x, overlap_y = 1672, 1307, 460, 460
            elif img_h == 2176:
                tile_size_x, tile_size_y, overlap_x, overlap_y = 1672, 1307, 460, 438

            # Find the corresponding label path
            if not args.images_only:
                if not args.list_file:
                    # Assumes 'labels' folder is directly under the argument's directory
                    labels_dir = os.path.join(os.path.abspath(args.directory), 'labels')
                    lbl_path = os.path.join(labels_dir, base_name + '.txt')
                else:
                    # List file mode: Find the label path by matching the image path index
                    try:
                        idx = image_paths.index(img_path)
                        lbl_path = label_paths[idx]
                    except ValueError:
                        print(f"Warning: Could not find label match for image {img_path}. Skipping labels for this image.")
                        lbl_path = None # Set to None to skip label processing later

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

                            # Example: Skip any clipped box that is smaller than 4 pixels in either dimension.
                            MIN_PIXEL_SIZE = 4 

                            # Calculate normalized min width/height (e.g., 2 pixels / 1672 = ~0.0012)
                            MIN_NORMALIZED_WIDTH = MIN_PIXEL_SIZE / tile_width_px
                            MIN_NORMALIZED_HEIGHT = MIN_PIXEL_SIZE / tile_height_px

                            # CRITICAL NEW FILTER: Skip if clipped width OR height is less than 50% of the original.
                            if clipped_width < (width * 0.5) or clipped_height < (height * 0.5):
                                continue

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
                            # Add the instability check:
                            if new_width < MIN_NORMALIZED_WIDTH or new_height < MIN_NORMALIZED_HEIGHT:
                                continue # Skip this box
                            
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