import glob
import os
import cv2
import argparse
import tqdm

def parse_arguments():
    """Parses command-line arguments for the tiling script."""
    parser = argparse.ArgumentParser(description="Tiling datasets for training")
    parser.add_argument('--source_dir', required=True, help="Directory containing images and/or labels folders.")
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
        with open(path, 'r', encoding='utf-8-sig') as f:
            path_list = f.read().splitlines()
        assert len(path_list) > 0, "No image paths found in csv"
    except FileNotFoundError:
        raise ValueError("Must provide an image directory or a path to a csv listing filepaths of images")
    return sorted(path_list)

def get_file_paths(args):
    """Retrieves file paths based on command-line arguments."""
    dir_path = os.path.abspath(args.directory)
    image_paths, label_paths = [], []
    
    # Get image paths
    if not args.labels_only:
        if args.list_file:
            image_paths = return_path_list(os.path.join(dir_path, "images.txt"))
        else:
            image_dir = os.path.join(dir_path, "images")
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff']:
                image_paths.extend(glob.glob(os.path.join(image_dir, ext)))
    
    # Get label paths
    if not args.images_only:
        if args.list_file:
            label_paths = return_path_list(os.path.join(dir_path, "labels.txt"))
        else:
            label_paths.extend(glob.glob(os.path.join(dir_path, "labels", '*.txt')))

    if not image_paths and not label_paths:
        raise ValueError("No image or label files found in the specified directory.")
        
    if not args.images_only and not args.labels_only and len(image_paths) != len(label_paths):
         raise ValueError("Mismatch between image and label counts.")
         
    return sorted(image_paths), sorted(label_paths)

def get_tile_params(img_h, args):
    """
    Returns tile dimensions and overlaps.
    Prioritizes hardcoded overrides for specific resolutions (legacy support),
    otherwise falls back to argparse defaults.
    """
    if img_h == 3000:
        return 1672, 1307, 460, 460
    elif img_h == 2176:
        return 1672, 1307, 460, 438
    else:
        return args.tile_size_x, args.tile_size_y, args.overlap_x, args.overlap_y

def filter_resume_paths(image_paths, label_paths, tiled_label_path, args):
    """
    Filters out paths that have already been processed if resume is enabled.
    """
    # If checking completion, we prefer the label directory as it's the last step in the loop
    check_dir = tiled_label_path if not args.images_only else os.path.join(os.path.dirname(tiled_label_path), 'images')
    
    if not os.path.exists(check_dir):
        return image_paths, label_paths

    tiles_complete = os.listdir(check_dir)
    if not tiles_complete:
        return image_paths, label_paths

    # Extract original basenames from tiled filenames (basename_x_y.ext)
    basenames_complete = {x.rsplit('_', 2)[0] for x in tiles_complete}
    
    # Remove the lexicographically last basename to ensure partial processing is redone
    if basenames_complete:
        last_processed = max(basenames_complete)
        basenames_complete.remove(last_processed)

    # Filter Image Paths
    if image_paths:
        original_map = {os.path.splitext(os.path.basename(p))[0]: p for p in image_paths}
        remaining_bases = set(original_map.keys()) - basenames_complete
        image_paths = sorted([original_map[b] for b in remaining_bases])

    # Filter Label Paths
    if label_paths:
        original_map = {os.path.splitext(os.path.basename(p))[0]: p for p in label_paths}
        remaining_bases = set(original_map.keys()) - basenames_complete
        label_paths = sorted([original_map[b] for b in remaining_bases])
        
    print(f"Resuming: {len(image_paths or label_paths)} items remaining.")
    return image_paths, label_paths

def get_clipped_labels(labels_data, tile_coords, img_dims):
    """
    Calculates new normalized labels for a specific tile.
    tile_coords: (x_start, y_start, x_end, y_end)
    img_dims: (img_w, img_h)
    """
    x_start, y_start, x_end, y_end = tile_coords
    img_w, img_h = img_dims
    tile_w_px = x_end - x_start
    tile_h_px = y_end - y_start
    
    new_labels = []
    
    for label in labels_data:
        class_id = int(label[0])
        # De-normalize YOLO coordinates to Absolute Pixels
        x_c, y_c, w, h = [float(label[i]) * dim for i, dim in enumerate([img_w, img_h, img_w, img_h], 1)]
        
        # Calculate Absolute Bounding Box
        x_min, y_min = x_c - w / 2, y_c - h / 2
        x_max, y_max = x_c + w / 2, y_c + h / 2
        
        # 1. Intersection Check
        if not (x_min < x_end and x_max > x_start and y_min < y_end and y_max > y_start):
            continue

        # 2. Clip at Tile Boundaries
        c_x_min = max(x_min, x_start)
        c_y_min = max(y_min, y_start)
        c_x_max = min(x_max, x_end)
        c_y_max = min(y_max, y_end)

        c_w = c_x_max - c_x_min
        c_h = c_y_max - c_y_min

        # 3. Filter: Box must retain at least 50% of original size
        if c_w < (w * 0.5) or c_h < (h * 0.5):
            continue

        # 4. Filter: Box must not be vanishingly small (e.g. < 4px)
        MIN_PIXEL_SIZE = 4
        if c_w < MIN_PIXEL_SIZE or c_h < MIN_PIXEL_SIZE:
            continue

        # 5. Normalize to Tile Dimensions
        new_cx = (c_x_min + c_w / 2 - x_start) / tile_w_px
        new_cy = (c_y_min + c_h / 2 - y_start) / tile_h_px
        new_w = c_w / tile_w_px
        new_h = c_h / tile_h_px
        
        new_labels.append(f"{class_id} {new_cx:.6f} {new_cy:.6f} {new_w:.6f} {new_h:.6f}")
        
    return new_labels

def main():
    args = parse_arguments()
    
    if args.labels_only and (args.image_width is None or args.image_height is None):
        raise ValueError("Error: --image_width and --image_height are required in --labels_only mode.")

    image_paths, label_paths = get_file_paths(args)

    base_dir = os.path.join(os.path.abspath(args.directory), "tiled")
    tiled_image_dir = os.path.join(base_dir, 'images')
    tiled_label_dir = os.path.join(base_dir, 'labels')

    if not args.labels_only: os.makedirs(tiled_image_dir, exist_ok=True)
    if not args.images_only: os.makedirs(tiled_label_dir, exist_ok=True)

    if args.resume:
        image_paths, label_paths = filter_resume_paths(image_paths, label_paths, tiled_label_dir, args)

    # Main Processing Loop
    paths_to_process = label_paths if args.labels_only else image_paths
    
    for file_path in tqdm.tqdm(paths_to_process):
        # 1. Setup Image/Label Data
        img, img_h, img_w = None, args.image_height, args.image_width
        labels_data = []
        
        if args.labels_only:
            lbl_path = file_path
            base_name = os.path.splitext(os.path.basename(lbl_path))[0]
        else:
            img_path = file_path
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Could not read {img_path}. Skipping.")
                continue
            img_h, img_w = img.shape[:2]
            
            if not args.images_only:
                # Resolve label path
                if args.list_file:
                    lbl_path = label_paths[image_paths.index(img_path)]
                else:
                    lbl_path = os.path.join(os.path.abspath(args.directory), 'labels', base_name + '.txt')

        # Load Label Data if needed
        if not args.images_only and lbl_path and os.path.exists(lbl_path):
            with open(lbl_path, 'r') as f:
                labels_data = [line.strip().split() for line in f.readlines()]

        # 2. Get Tile Config (re-calculated per image to prevent state leakage)
        ts_x, ts_y, ov_x, ov_y = get_tile_params(img_h, args)

        # 3. Perform Tiling
        y = 0
        while True:
            y_start = y
            y_end = min(y + ts_y, img_h)
            if y_end == img_h: y_start = img_h - ts_y # Snap to bottom edge

            x = 0
            while True:
                x_start = x
                x_end = min(x + ts_x, img_w)
                if x_end == img_w: x_start = img_w - ts_x # Snap to right edge

                # --- Save Image Tile ---
                if not args.labels_only:
                    tile = img[y_start:y_end, x_start:x_end]
                    cv2.imwrite(os.path.join(tiled_image_dir, f"{base_name}_{x_start}_{y_start}.png"), tile)

                # --- Save Label Tile ---
                if not args.images_only:
                    tile_coords = (x_start, y_start, x_end, y_end)
                    new_labels = get_clipped_labels(labels_data, tile_coords, (img_w, img_h))
                    
                    with open(os.path.join(tiled_label_dir, f"{base_name}_{x_start}_{y_start}.txt"), 'w') as f:
                        f.write('\n'.join(new_labels))

                # Loop Control
                if x_end == img_w: break
                x += ts_x - ov_x
            
            if y_end == img_h: break
            y += ts_y - ov_y

if __name__ == '__main__':
    main()