import os
import cv2
import numpy as np
import argparse
import tqdm
import glob
from collections import defaultdict

def parse_arguments():
    parser = argparse.ArgumentParser(description="Reconstruct (Untile) datasets from tiles")
    parser.add_argument('--directory', required=True, help="Directory containing the 'tiled' folder")
    parser.add_argument('--output_dir', default=None, help="Directory to save reconstructed images/labels.")
    parser.add_argument('--image_width', type=int, default=None, help="Force original image width")
    parser.add_argument('--image_height', type=int, default=None, help="Force original image height")
    parser.add_argument('--tile_size_x', type=int, default=1672)
    parser.add_argument('--tile_size_y', type=int, default=1307)
    # Threshold for merging
    parser.add_argument('--nms_threshold', type=float, default=0.5, help="IoM threshold to merge boxes.")
    
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--images_only', action="store_true", help='Only reconstruct images.')
    group.add_argument('--labels_only', action="store_true", help="Only reconstruct labels.")
    
    return parser.parse_args()

def parse_tile_filename(filename):
    name_no_ext = os.path.splitext(os.path.basename(filename))[0]
    try:
        parts = name_no_ext.rsplit('_', 2)
        base_name = parts[0]
        x_start = int(parts[1])
        y_start = int(parts[2])
        return base_name, x_start, y_start
    except (IndexError, ValueError):
        return None, None, None

def get_grouped_files(source_dir, subfolder, valid_exts):
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
            if base not in grouped: grouped[base] = []
            grouped[base].append(f)
    return grouped

def calculate_canvas_size(tile_paths, tile_w, tile_h, forced_w=None, forced_h=None):
    if forced_w and forced_h: return forced_w, forced_h
    max_x, max_y = 0, 0
    for p in tile_paths:
        _, x, y = parse_tile_filename(p)
        max_x = max(max_x, x)
        max_y = max(max_y, y)
    return forced_w if forced_w else max_x + tile_w, forced_h if forced_h else max_y + tile_h

def reconstruct_images(grouped_images, output_dir, args):
    dest_dir = os.path.join(output_dir, "images")
    os.makedirs(dest_dir, exist_ok=True)
    print(f"Reconstructing {len(grouped_images)} images...")
    
    for base_name, paths in tqdm.tqdm(grouped_images.items()):
        W, H = calculate_canvas_size(paths, args.tile_size_x, args.tile_size_y, args.image_width, args.image_height)
        canvas = np.zeros((H, W, 3), dtype=np.uint8)
        
        for tile_path in paths:
            _, x_start, y_start = parse_tile_filename(tile_path)
            tile = cv2.imread(tile_path)
            if tile is None: continue
            h_t, w_t, _ = tile.shape
            y_end = min(y_start + h_t, H)
            x_end = min(x_start + w_t, W)
            t_h = y_end - y_start
            t_w = x_end - x_start
            if t_h > 0 and t_w > 0:
                canvas[y_start:y_end, x_start:x_end] = tile[:t_h, :t_w]
        cv2.imwrite(os.path.join(dest_dir, f"{base_name}.png"), canvas)

def compute_iom(box1, box2):
    """
    Computes Intersection over Minimum (IoM).
    This handles cases where a small box (fragment) is completely inside a larger box.
    """
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0: return 0.0
    
    boxAArea = (box1[2] - box1[0]) * (box1[3] - box1[1])
    boxBArea = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # Use MIN area to ensure containment = 1.0 score
    return interArea / float(min(boxAArea, boxBArea))

def merge_boxes_union(boxes, class_ids, threshold):
    if not boxes: return [], []
    
    # Group by class
    by_class = defaultdict(list)
    for i, cls in enumerate(class_ids):
        by_class[cls].append(i)
    
    final_boxes = []
    final_classes = []
    
    for cls, indices in by_class.items():
        cls_boxes = [boxes[i] for i in indices]
        # Convert to x1,y1,x2,y2
        cls_boxes_xyxy = [[b[0], b[1], b[0]+b[2], b[1]+b[3]] for b in cls_boxes]
        
        n = len(cls_boxes_xyxy)
        parent = list(range(n))
        
        def find(i):
            if parent[i] != i: parent[i] = find(parent[i])
            return parent[i]
        
        def union(i, j):
            rootA = find(i)
            rootB = find(j)
            if rootA != rootB: parent[rootB] = rootA
            
        # Check overlaps
        for i in range(n):
            for j in range(i + 1, n):
                if compute_iom(cls_boxes_xyxy[i], cls_boxes_xyxy[j]) > threshold:
                    union(i, j)
        
        # Group components
        groups = defaultdict(list)
        for i in range(n):
            groups[find(i)].append(cls_boxes_xyxy[i])
            
        # Merge
        for group in groups.values():
            min_x = min(b[0] for b in group)
            min_y = min(b[1] for b in group)
            max_x = max(b[2] for b in group)
            max_y = max(b[3] for b in group)
            final_boxes.append([min_x, min_y, max_x - min_x, max_y - min_y])
            final_classes.append(cls)
            
    return final_boxes, final_classes

def reconstruct_labels(grouped_labels, output_dir, args, grouped_images_ref=None):
    dest_dir = os.path.join(output_dir, "labels")
    os.makedirs(dest_dir, exist_ok=True)
    print(f"Reconstructing {len(grouped_labels)} labels...")
    
    for base_name, paths in tqdm.tqdm(grouped_labels.items()):
        ref_paths = grouped_images_ref.get(base_name, paths) if grouped_images_ref else paths
        W, H = calculate_canvas_size(ref_paths, args.tile_size_x, args.tile_size_y, args.image_width, args.image_height)
        
        all_boxes = []
        all_class_ids = []
        
        for lbl_path in paths:
            _, x_off, y_off = parse_tile_filename(lbl_path)
            with open(lbl_path, 'r') as f:
                for line in f:
                    parts = list(map(float, line.strip().split()))
                    if len(parts) < 5: continue
                    cls_id = int(parts[0])
                    cx, cy, w, h = parts[1:]
                    
                    cur_tw = min(args.tile_size_x, W - x_off)
                    cur_th = min(args.tile_size_y, H - y_off)
                    
                    # To pixels
                    w_pix = w * cur_tw
                    h_pix = h * cur_th
                    x1 = (cx * cur_tw) + x_off - (w_pix / 2)
                    y1 = (cy * cur_th) + y_off - (h_pix / 2)
                    
                    all_boxes.append([int(x1), int(y1), int(w_pix), int(h_pix)])
                    all_class_ids.append(cls_id)
        
        if not all_boxes:
            open(os.path.join(dest_dir, f"{base_name}.txt"), 'w').close()
            continue
            
        final_boxes, final_classes = merge_boxes_union(all_boxes, all_class_ids, args.nms_threshold)
        
        lines = []
        for i, (x, y, w, h) in enumerate(final_boxes):
            cx = (x + w/2) / W
            cy = (y + h/2) / H
            nw = w / W
            nh = h / H
            lines.append(f"{final_classes[i]} {min(max(cx,0),1):.6f} {min(max(cy,0),1):.6f} {min(max(nw,0),1):.6f} {min(max(nh,0),1):.6f}")
            
        with open(os.path.join(dest_dir, f"{base_name}.txt"), 'w') as f:
            f.write('\n'.join(lines))

def main():
    args = parse_arguments()
    output_dir = args.output_dir if args.output_dir else os.path.join(os.path.abspath(args.directory), "untiled")
    
    if os.path.basename(os.path.abspath(args.directory)) in ['images', 'labels']:
        base = os.path.dirname(os.path.abspath(args.directory))
    else:
        base = os.path.abspath(args.directory)

    g_imgs = {}
    g_lbls = {}
    
    if not args.labels_only:
        g_imgs = get_grouped_files(base, "images", ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff'])
    if not args.images_only:
        g_lbls = get_grouped_files(base, "labels", ['*.txt'])

    if not g_imgs and not g_lbls:
        print("Error: No files found.")
        return

    if not args.labels_only and g_imgs: reconstruct_images(g_imgs, output_dir, args)
    if not args.images_only and g_lbls: reconstruct_labels(g_lbls, output_dir, args, g_imgs)
    
    print("Reconstruction complete.")

if __name__ == '__main__':
    main()