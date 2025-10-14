import os
import glob
from typing import List
from tqdm import tqdm # Import tqdm

# --- Configuration ---
# 1. Define the base directory for your labels
BASE_LABEL_DIR = r"Z:\__Organized_Directories_InProgress\GobyFinderDatasets\AUV_datasets\full\labels" # Adjust this path to your actual label directory

# 2. Define the file extension for your label files
LABEL_EXTENSION = ".txt"

def remove_duplicates_from_label_file(label_path: str) -> int:
    """Reads a label file, removes duplicate lines, and overwrites the file."""
    
    # 1. Read all lines from the file
    try:
        with open(label_path, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        # We can't use 'tqdm.write' here as the function is run inside the loop
        # and doesn't know the tqdm instance. We'll handle errors in the main block.
        raise FileNotFoundError(f"Label file not found: {label_path}")

    # 2. Use a set to get unique lines while preserving order (Python 3.7+ friendly)
    unique_lines = list(dict.fromkeys(lines)) 
    
    removed_count = len(lines) - len(unique_lines)

    # 3. Overwrite the file with only the unique lines
    if removed_count > 0:
        with open(label_path, 'w') as f:
            f.writelines(unique_lines)

    return removed_count

# ----------------- Main Logic with TQDM -----------------
if __name__ == "__main__":
    # 1. Construct the pattern to find all label files
    search_pattern = os.path.join(BASE_LABEL_DIR, f"*{LABEL_EXTENSION}")
    
    # 2. Find all matching label files
    all_label_files: List[str] = glob.glob(search_pattern)

    if not all_label_files:
        print(f"No label files found in: {BASE_LABEL_DIR} with pattern {LABEL_EXTENSION}")
    else:
        total_removed = 0
        
        # 3. Wrap the list of files with tqdm()
        # The 'desc' argument sets the label for the progress bar
        print(f"Starting duplicate removal across {len(all_label_files)} label files...")
        
        for label_file in tqdm(all_label_files, desc="Processing Labels"):
            try:
                removed_count = remove_duplicates_from_label_file(label_file)
                total_removed += removed_count
                
                # Use tqdm.write() to print messages without disrupting the bar
                if removed_count > 0:
                    tqdm.write(f"  Removed {removed_count} duplicates from: {os.path.basename(label_file)}")
                    
            except FileNotFoundError as e:
                tqdm.write(f"ERROR: {e}")
            
        print("\n--- Summary ---")
        print(f"Total duplicate labels removed across all files: {total_removed}")
        print("Processing complete.")