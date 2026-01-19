import os
import shutil  # Added for the backup feature
import pandas as pd
import sys
import numpy as np
import pathlib
SCRIPT_DIR = pathlib.Path(__file__).parent if '__file__' in locals() else pathlib.Path.cwd()
sys.path.append(str(SCRIPT_DIR.parent.parent / "src"))
from utils import Utils
from reportFunctions import Reports
# -----------------------------------------------
def lbl_report_filt_min_pixels(df, min_pixels=4):
    """
    Identifies labels whose normalized width or height corresponds to fewer than 
    MIN_PIXELS in the original tile.
    
    Assumption: Since the training size is 1672, we use 1672 as the maximum 
    possible dimension for the normalized conversion.
    """
    # Use the max tile dimension (width or square input size) for a conservative pixel conversion
    MAX_TILE_DIM = 1672
    
    # Calculate the minimum normalized value required
    min_normalized = min_pixels / MAX_TILE_DIM
    
    # Filter for boxes where the normalized width OR height is below the threshold
    # Note: df.w and df.h are normalized to the tile size
    # If a tile is 1672x1307, the normalized w=0.0024 is 4px, but h=0.003 is 4px.
    # Using 1672 for both checks safely removes objects that are too small 
    # in the 1307 dimension as well.
    return df[(df.w < min_normalized) | (df.h < min_normalized)]

def lbl_report_filt_g(df):
    """
    Your provided filter function.
    Finds labels with weights < 0.1 or > 80.
    """
    return df[((df.box_DL_weight_g_corr < 0.1) | (df.box_DL_weight_g_corr > 80)) & (df.conf.notna())]

def filter_yolo_lbl_report_and_remove_labels(
    run_directory, 
    labels_directory, 
    save_new_labels_csv=True, 
    save_new_labels=True,
    backup_originals=True  # <-- SAFETY: Added backup flag
):
    """
    Filters a master label report, saves a cleaned version, and then
    iterates through individual YOLO .txt files to remove the filtered
    objects, creating backups first.
    """
    
    # --- 1. Setup Paths ---
    meta_path = os.path.join(run_directory, "metadata.csv") # User updated this
    yolo_lbl_path = os.path.join(run_directory, "labels_edit.csv")
    substrate_path = None # As in your original code
    op_path = r"Z:\__AdvancedTechnologyBackup\07_Database\OP_TABLE.xlsx"
    
    # --- 2. Load and Filter Main Report ---
    print("Loading label report...")
    if not os.path.exists(yolo_lbl_path):
        print(f"ERROR: Cannot find labels.csv at: {yolo_lbl_path}")
        return
        
    # This is your external call, we assume it works
    label_report = Reports.output_LBL_results(meta_path, yolo_lbl_path, substrate_path, op_path)
    
    lbl_df = pd.read_csv(yolo_lbl_path, index_col=0)
    print(f"Loaded labels.csv with {len(lbl_df)} total objects.")

    # 🎯 APPLY AND COMBINE BOTH FILTERS 🎯
    
    # Filter A: Original Filter (Weight)
    filt_tiles_weight = lbl_report_filt_g(label_report)
    
    # Filter B: New Filter (Min Pixel Size = 4)
    filt_tiles_pixels = lbl_report_filt_min_pixels(label_report, min_pixels=4)
    
    # Combine ground_truth_ids from both filters and remove duplicates
    filt_tiles_ground_truth = pd.concat([
        filt_tiles_weight.ground_truth_id, 
        filt_tiles_pixels.ground_truth_id
    ]).drop_duplicates().reset_index(drop=True)
    
    assert not filt_tiles_ground_truth.duplicated().any(), "Duplicate ground_truth_ids found in filter list"
    print(f"Found {len(filt_tiles_ground_truth)} objects to remove based on filtering.")

    if len(filt_tiles_ground_truth) == 0:
        print("No objects to remove. Exiting.")
        return

    # --- 3. Create Cleaned Master CSV (Good for logging/backup) ---
    tile_lbl_edit = lbl_df[~lbl_df.ground_truth_id.isin(filt_tiles_ground_truth)]
    
    # Your excellent sanity check
    # Calculate the total number of *rows* in lbl_df that match the unique bad IDs
    rows_to_remove_count = len(lbl_df[lbl_df.ground_truth_id.isin(filt_tiles_ground_truth)])
    
    # Filter the master report
    tile_lbl_edit = lbl_df[~lbl_df.ground_truth_id.isin(filt_tiles_ground_truth)]
    
    # Corrected sanity check: 
    # (Original Rows) - (Actual Rows Removed) == (Remaining Rows)
    assert len(lbl_df) - rows_to_remove_count == len(tile_lbl_edit), \
        "Sanity check failed: The number of rows removed from the master report does not match the expected count."
    
    if save_new_labels_csv:
        edit_path = os.path.join(run_directory, "labels_edit.csv")
        tile_lbl_edit.to_csv(edit_path)
        print(f"Saved cleaned master list to {edit_path}")

    # --- 4. Identify Objects and Files to Modify ---
    # This DataFrame contains *only* the rows to be removed
    tile_lbl_remove_object = lbl_df[lbl_df.ground_truth_id.isin(filt_tiles_ground_truth)]
    
    # **REFACTORED LOGIC**
    # Group the objects-to-remove by their source filename.
    grouped_removals = tile_lbl_remove_object.groupby('Filename')
    
    print(f"\nFound {len(grouped_removals)} unique .txt files to modify and {len(tile_lbl_remove_object)} objects to drop.")

    # --- 5. Loop, Backup, and Overwrite ---
    for basename, group_df in grouped_removals:
        # 'basename' is the Filename (e.g., "PI_1596969633_...")
        # 'group_df' contains the rows from lbl_df for just that file
        
        # the unique cls,x,y,w,h to remove
        drop_df = group_df[["cls","x","y","w","h"]]
        lbl_file_path = os.path.join(labels_directory, basename + ".txt")
        
        print(f"Processing: {basename}.txt", end=' \r')

        if not os.path.exists(lbl_file_path):
            print(f"  ...WARNING: File not found, skipping: {lbl_file_path}", end=" \r")
            continue
            
        try:
            # **SAFETY FEATURE** (User's new logic)
            if backup_originals:
                backup_path = lbl_file_path + ".bak"
                if os.path.exists(backup_path):
                    print("  ...backup already exists, not overwriting.", end=' \r')
                else:
                    shutil.copy2(lbl_file_path, backup_path)
                    print(f"  ...Backed up original to: {backup_path}", end=' \r')
                    
            # Read the original .txt file
            lbl = Utils.read_YOLO_lbl(lbl_file_path)
            if lbl.empty:
                 print("  ...File is empty, skipping.", end=" \r")
                 continue
            
            # **ROBUST FILTERING LOGIC**
            # We must round both DataFrames to handle float precision issues.
            precision = 6 # Standard YOLOv8 precision
            
            lbl_rounded = lbl.round(precision)
            # De-duplicate the "drop" list. We want to remove ALL instances
            # of an object, even if it appears multiple times in the file.
            drop_df_unique_rounded = drop_df.round(precision).drop_duplicates()

            if drop_df_unique_rounded.empty:
                print("  ...No valid objects to remove for this file.", end=" \r")
                continue

            # Perform an "anti-join" using a left merge with an indicator.
            # This finds all rows in 'lbl' that *do not* have a match in 'drop_df'.
            merged = pd.merge(
                lbl_rounded, 
                drop_df_unique_rounded, 
                on=["cls","x","y","w","h"], 
                how='left', 
                indicator=True
            )
            
            # Keep only the rows that were 'left_only' (i.e., not in drop_df)
            # We use the *original* 'lbl' DataFrame to preserve original precision
            lbl_cleaned = lbl[merged['_merge'] == 'left_only']
            
            num_removed = len(lbl) - len(lbl_cleaned)
            if num_removed == 0:
                print("  ...WARNING: No matching objects found to remove.", end=" \r")
                continue # No need to save if nothing changed

            # Overwrite the original file
            if save_new_labels:
                Utils.save_YOLO_lbl(lbl_cleaned, lbl_file_path)
                print(f"  ...Successfully cleaned and saved ({num_removed} removed).", end=" \r")
            
        except Exception as e:
            print(f"\n  ...ERROR processing file {lbl_file_path}: {e}")
            # Optionally, you could restore the backup here
            # if os.path.exists(backup_path):
            #     shutil.move(backup_path, lbl_file_path)
            
    print("\nLabel cleaning process complete.")

# --- Example of how to run this ---
if __name__ == "__main__":
    # --- (User's real directories) ---
    run_dir = r"D:\ageglio-1\gobyfinder_yolov8\output\test_runs\Labeled data tiled 2048 HNM"
    tiles_dir = r"D:\datasets\Extra\full\labels"
    
    print("--- Running Cleaner ---")
    filter_yolo_lbl_report_and_remove_labels(run_dir, tiles_dir, save_new_labels_csv=False, save_new_labels=False)
    print("--- Cleaner Finished ---")

