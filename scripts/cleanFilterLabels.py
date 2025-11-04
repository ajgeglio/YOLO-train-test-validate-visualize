import os
import shutil  # Added for the backup feature
import pandas as pd
import sys
import numpy as np
# --- Use Actual Classes ---
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from utils import Utils
from reports import Reports
# -----------------------------------------------


def lbl_report_filt_g(df):
    """
    Your provided filter function.
    Finds labels with weights < 0.1 or > 80.
    """
    return df[((df.box_DL_weight_g_corr < 0.1) | (df.box_DL_weight_g_corr > 80)) & (df.conf.notna())]

def filter_yolo_lbl_report_and_remove_labels(
    run_directory, 
    tiles_directory, 
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
    yolo_lbl_path = os.path.join(run_directory, "labels_orig.csv")
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
    
    filt_tiles = lbl_report_filt_g(label_report)
    filt_tiles_ground_truth = filt_tiles.ground_truth_id
    
    assert not filt_tiles_ground_truth.duplicated().any(), "Duplicate ground_truth_ids found in filter list"
    print(f"Found {len(filt_tiles_ground_truth)} objects to remove based on filtering.")

    if len(filt_tiles_ground_truth) == 0:
        print("No objects to remove. Exiting.")
        return

    # --- 3. Create Cleaned Master CSV (Good for logging/backup) ---
    tile_lbl_edit = lbl_df[~lbl_df.ground_truth_id.isin(filt_tiles_ground_truth)]
    
    # Your excellent sanity check
    assert len(lbl_df) - len(filt_tiles_ground_truth) == len(tile_lbl_edit)
    
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
        lbl_file_path = os.path.join(tiles_directory, basename + ".txt")
        
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
            print(f"  ...ERROR processing file {lbl_file_path}: {e}")
            # Optionally, you could restore the backup here
            # if os.path.exists(backup_path):
            #     shutil.move(backup_path, lbl_file_path)
            
    print("\nLabel cleaning process complete.")

# --- Example of how to run this ---
if __name__ == "__main__":
    # --- (User's real directories) ---
    run_dir = r"D:\ageglio-1\gobyfinder_yolov8\output\test_runs\Labeled data 2048 All Run13"
    tiles_dir = r"D:\datasets\tiled\train\images"
    
    print("--- Running Cleaner ---")
    filter_yolo_lbl_report_and_remove_labels(run_dir, tiles_dir, save_new_labels_csv=False, save_new_labels=False)
    print("--- Cleaner Finished ---")

