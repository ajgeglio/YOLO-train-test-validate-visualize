import os
import cv2
import numpy as np
import argparse
import tqdm
import glob

def parse_arguments():
    """Parses command-line arguments for the untilling script."""
    parser = argparse.ArgumentParser(description="Reconstruct (Untile) datasets from tiles")
    parser.add_argument('--source_dir', required=True, help="Directory containing the 'tiled' folder (or path to tiled/images and tiled/labels).")
    parser.add_argument('--output_dir', default=None, help="Directory to save reconstructed images/labels.")
    
    # Dimensions are optional; if not provided, script calculates max extents from tile coordinates
    parser.add_argument('--image_width', type=int, default=None, help="Force original image width (optional, auto-detected if None).")
    parser.add_argument('--image_height', type=int, default=None, help="Force original image height (optional, auto-detected if None).")
    
    # Tile dimensions must match the settings used during tiling
    parser.add_argument('--tile_size_x', type=int, default=1672)
    parser.add_argument('--tile_size_y', type=int, default=1307)
    
    # NMS Threshold for merging duplicate labels in overlap regions
    parser.add_argument('--nms_threshold', type=float, default=0.5, help="IoU threshold to merge overlapping boxes (0.0 to 1.0).")
    
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--images_only', action="store_true", help='Only reconstruct images.')
    group.add_argument('--labels_only', action="store_true", help="Only reconstruct labels.")
    
    return parser.parse_args()

def parse_tile_filename(filename):
    """
    Extracts base_name, x_offset, and y_offset from 'basename_x_y.ext'.
    Returns: (base_name, x, y)
    """
    name_no_ext = os.path.splitext(os.path.basename(filename))[0]
    try:
        # Split from the right to handle basenames that might contain underscores
        parts = name_no_ext.rsplit('_', 2)
        base_name = parts[0]
        x_start = int(parts[1])
        y_start = int(parts[2])
        return base_name, x_start, y_start
    except (IndexError, ValueError):
        print(f"Warning: Could not parse coordinates from {filename}. Skipping.")
        return None, None, None

def get_grouped_files(source_dir, subfolder, valid_exts):
    """
    Scans a directory and groups tile paths by their original basename.
    Returns: dict { 'basename': [path1, path2, ...] }
    """
    search_path = os.path.join(source_dir, subfolder)
    if not os.path.exists(search_path):
        return {}
        
    files = []
    for ext in valid_exts:
        files.extend(glob.glob(os.path.join(search_path, ext)))
    
    grouped = {}
    for f in files:
        base, _, _ = parse_tile_filename(f)
        if base:
            if base not in grouped:
                grouped[base] = []
            grouped[base].append(f)
    return grouped

def calculate_canvas_size(tile_paths, tile_w, tile_h, forced_w=None, forced_h=None):
    """
    Determines the required canvas size to fit all tiles.
    """
    if forced_w and forced_h:
        return forced_w, forced_h

    max_x, max_y = 0, 0
    for p in tile_paths:
        _, x, y = parse_tile_filename(p)
        max_x = max(max_x, x)
        max_y = max(max_y, y)
    
    # The canvas must end at the last pixel of the furthest tile
    # Note: Tiling script clamps edges, so max_x + tile_w might exceed original slightly 
    # if not perfectly aligned, but standard OpenCV cropping handles out-of-bounds gracefully.
    # We essentially want the bounding box of all tiles.
    calculated_w = max_x + tile_w
    calculated_h = max_y + tile_h
    
    return forced_w if forced_w else calculated_w, forced_h if forced_h else calculated_h

def reconstruct_images(grouped_images, output_dir, args):
    """
    Stitches image tiles back together.
    """
    dest_dir = os.path.join(output_dir, "images")
    os.makedirs(dest_dir, exist_ok=True)
    
    print(f"Reconstructing {len(grouped_images)} images...")
    
    for base_name, paths in tqdm.tqdm(grouped_images.items()):
        # Determine Canvas Size
        W, H = calculate_canvas_size(paths, args.tile_size_x, args.tile_size_y, args.image_width, args.image_height)
        
        # Create black canvas
        canvas = np.zeros((H, W, 3), dtype=np.uint8)
        
        for tile_path in paths:
            _, x_start, y_start = parse_tile_filename(tile_path)
            tile = cv2.imread(tile_path)
            if tile is None:
                continue
                
            h_t, w_t, _ = tile.shape
            
            # Paste tile. 
            # Note: Later tiles overwrite earlier ones in overlap regions. 
            # Since source pixels are identical, order doesn't matter.
            
            # Handle edge cases where canvas might be smaller than calculated (if user forced dims)
            y_end = min(y_start + h_t, H)
            x_end = min(x_start + w_t, W)
            
            # Crop tile if it goes out of forced bounds
            t_h_crop = y_end - y_start
            t_w_crop = x_end - x_start
            
            if t_h_crop > 0 and t_w_crop > 0:
                canvas[y_start:y_end, x_start:x_end] = tile[:t_h_crop, :t_w_crop]
                
        cv2.imwrite(os.path.join(dest_dir, f"{base_name}.png"), canvas)

def reconstruct_labels(grouped_labels, output_dir, args, grouped_images_ref=None):
    """
    Stitches labels back together and applies NMS to remove duplicates.
    grouped_images_ref: used to look up dimensions if args.image_width/height are missing.
    """
    dest_dir = os.path.join(output_dir, "labels")
    os.makedirs(dest_dir, exist_ok=True)
    
    print(f"Reconstructing {len(grouped_labels)} labels...")
    
    for base_name, paths in tqdm.tqdm(grouped_labels.items()):
        # 1. Determine Canvas Size (needed for normalization)
        # Try to infer from images if available (more accurate), else from label tiles
        ref_paths = grouped_images_ref.get(base_name, paths) if grouped_images_ref else paths
        W, H = calculate_canvas_size(ref_paths, args.tile_size_x, args.tile_size_y, args.image_width, args.image_height)
        
        all_boxes = []
        all_class_ids = []
        all_scores = [] # YOLO doesn't have scores in GT, so we assign 1.0
        
        # 2. Collect all boxes from all tiles
        for lbl_path in paths:
            _, x_offset, y_offset = parse_tile_filename(lbl_path)
            
            with open(lbl_path, 'r') as f:
                lines = f.readlines()
                
            for line in lines:
                parts = list(map(float, line.strip().split()))
                if len(parts) < 5: continue
                
                cls_id = int(parts[0])
                cx_rel, cy_rel, w_rel, h_rel = parts[1:]
                
                # Denormalize relative to TILE dimensions
                # Note: We must use the specific tile size defined in args, 
                # effectively assuming the tile was args.tile_size.
                # If the tile was on the edge, the tiling script still normalized relative to the *clip*,
                # but let's check the tiling logic: 
                # Tiling: "new_x_center = (clipped_x_center - x_start) / tile_width_px"
                # Here tile_width_px is the ACTUAL width of the tile image (which might be smaller at edges).
                # To be precise, we need the actual tile width/height.
                # However, without reading the image, we assume the tile is the standard size 
                # unless x_offset + tile_size > W.
                
                current_tile_w = min(args.tile_size_x, W - x_offset)
                current_tile_h = min(args.tile_size_y, H - y_offset)
                
                # Recover pixel coordinates relative to the tile
                cx_pix = cx_rel * current_tile_w
                cy_pix = cy_rel * current_tile_h
                w_pix = w_rel * current_tile_w
                h_pix = h_rel * current_tile_h
                
                # Recover absolute global pixel coordinates
                global_cx = cx_pix + x_offset
                global_cy = cy_pix + y_offset
                
                # Convert to XYXY for NMS (TopLeft, BottomRight)
                x1 = global_cx - (w_pix / 2)
                y1 = global_cy - (h_pix / 2)
                # w_pix and h_pix are already calculated
                
                all_boxes.append([int(x1), int(y1), int(w_pix), int(h_pix)])
                all_class_ids.append(cls_id)
                all_scores.append(1.0) # Ground truth confidence
                
        if not all_boxes:
            # Create empty file
            open(os.path.join(dest_dir, f"{base_name}.txt"), 'w').close()
            continue
            
        # 3. Apply NMS (Non-Maximum Suppression)
        # This merges boxes that overlap significantly (duplicates from tile overlaps)
        indices = cv2.dnn.NMSBoxes(all_boxes, all_scores, score_threshold=0.5, nms_threshold=args.nms_threshold)
        
        final_lines = []
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = all_boxes[i]
                cls = all_class_ids[i]
                
                # 4. Re-normalize to Global Canvas Size
                # YOLO format: class cx cy w h (normalized 0-1)
                
                final_cx = (x + w / 2) / W
                final_cy = (y + h / 2) / H
                final_w = w / W
                final_h = h / H
                
                # Clamp to 0-1 just in case
                final_cx = min(max(final_cx, 0), 1)
                final_cy = min(max(final_cy, 0), 1)
                final_w = min(max(final_w, 0), 1)
                final_h = min(max(final_h, 0), 1)
                
                final_lines.append(f"{cls} {final_cx:.6f} {final_cy:.6f} {final_w:.6f} {final_h:.6f}")
        
        with open(os.path.join(dest_dir, f"{base_name}.txt"), 'w') as f:
            f.write('\n'.join(final_lines))

def main():
    args = parse_arguments()

    output_dir = args.output_dir if args.output_dir is not None else os.path.join(os.path.abspath(args.source_dir),"untiled")
    
    # Locate Tiled Directories
    # Support both "parent dir" inputs and specific subfolder inputs logic
    if os.path.basename(os.path.abspath(args.source_dir)) in ['images', 'labels']:
        # User pointed to .../tiled/images
        base_source = os.path.dirname(os.path.abspath(args.source_dir))
    else:
        # User pointed to .../tiled/
        base_source = os.path.abspath(args.source_dir)

    grouped_images = {}
    grouped_labels = {}

    # Scan for Images
    if not args.labels_only:
        grouped_images = get_grouped_files(base_source, "images", ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff'])
        
    # Scan for Labels
    if not args.images_only:
        grouped_labels = get_grouped_files(base_source, "labels", ['*.txt'])

    if not grouped_images and not grouped_labels:
        print("Error: No files found in the specified source directory.")
        return

    # Processing
    if not args.labels_only and grouped_images:
        reconstruct_images(grouped_images, output_dir, args)
        
    if not args.images_only and grouped_labels:
        # Pass grouped_images to help infer dimensions if available
        reconstruct_labels(grouped_labels, output_dir, args, grouped_images)

    print("Reconstruction complete.")

if __name__ == '__main__':
    main()