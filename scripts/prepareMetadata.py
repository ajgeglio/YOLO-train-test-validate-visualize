import argparse
import pathlib
from datetime import datetime
import pandas as pd
import numpy as np
import glob
import sys
import os
from typing import Optional, List, Any

from polars import col
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from utils import Utils

# --- Argument Parsing ---
def parse_args():
    """
    Parses command-line arguments for the metadata analysis script. 
    Assumes the metadata folder contains an updated combined metadata pickle file.
    """
    parser = argparse.ArgumentParser(description="Filter and clean all image metadata to get usable GOBY-assessed collects for a specified year.")
    
    # Required Paths
    parser.add_argument("--metadata_folder", type=pathlib.Path, 
        default=r"Z:\__AdvancedTechnologyBackup\07_Database\MetadataCombined",
        help="Path to the directory containing all_unpacked_images_metadata*.pkl/pickle files."
    )
    parser.add_argument("--output_folder", type=pathlib.Path, 
        default=r"Z:\__AdvancedTechnologyBackup\07_Database\MetadataCombined",
        help="Path to the directory where the processed metadata CSV will be saved."
    )
    parser.add_argument("--op_table_pth", type=pathlib.Path, 
        default=r"Z:\__AdvancedTechnologyBackup\07_Database\OP_TABLE.xlsx",
        help="Path to the operational assessment table (OP_TABLE.xlsx)."
    )
    # Image Directory (Used by --image_list_filter)
    parser.add_argument("--img_directory", type=pathlib.Path, required=False,
        help="Path to the images directory. Used to generate a list of image filenames for filtering."
    )
    parser.add_argument("--tiled", action="store_true",
        help="Metadata must be formatted for a tiled images"
    )
    # Mutually Exclusive Filtering Options
    group = parser.add_mutually_exclusive_group(required=True) # Made group required
    group.add_argument("--image_list_filter", action="store_true",
        help="Filter metadata based on filenames found in --img_directory."
    )
    group.add_argument("--goby_collects_filter", action="store_true",
        help="Filter metadata to include only 'GOBY == 1' assessed collects from the OP table."
    )
    group.add_argument("--collect_id", type=str,
        help="Specific COLLECT_ID to filter the metadata."
    )
    
    return parser.parse_args()

# --- Data Loading and Filtering ---
def load_metadata(args: argparse.Namespace) -> pd.DataFrame:
    """Loads the latest combined metadata pickle file from the specified folder."""
    if not args.metadata_folder.exists():
        print(f"Error: Metadata folder not found at {args.metadata_folder}, exiting.")
        sys.exit(1)

    print(f"Loading metadata from: {args.metadata_folder}")
    
    # Glob for both .pkl and .pickle files, and sort to find the latest one
    metadata_files = list(args.metadata_folder.glob("all_unpacked_images_metadata*.pkl"))
    metadata_files.extend(list(args.metadata_folder.glob("all_unpacked_images_metadata*.pickle")))

    if not metadata_files:
        print(f"No metadata files found in {args.metadata_folder}, exiting.")
        sys.exit(1)

    # Sort files lexicographically (assuming YYYYmmdd dates in filename)
    metadata_files.sort(reverse=True)
    latest_metadata_file = metadata_files[0]
    
    print(f"Loading latest metadata file: {latest_metadata_file}")

    try:
        metadata = pd.read_pickle(latest_metadata_file)
        # Convert collect_id to string for consistent filtering
        if 'collect_id' in metadata.columns:
             metadata['collect_id'] = metadata['collect_id'].astype(str)
        return metadata
    except Exception as e:
        print(f"Error loading metadata pickle file: {e}")
        sys.exit(1)

def filter_goby_collects(op_table_pth: pathlib.Path) -> pd.Series:
    """
    Reads the OP table and returns a Series of all COLLECT_IDs that are flagged 
    for GOBY assessments (GOBY == 1).
    """
    try:
        op_table = pd.read_excel(op_table_pth)
    except FileNotFoundError:
        print(f"Error: OP table not found at {op_table_pth}")
        sys.exit(1)
        
    # Ensure COLLECT_ID is string for consistent filtering
    op_table['COLLECT_ID'] = op_table['COLLECT_ID'].astype(str)
    op_table_assmt = op_table[op_table.GOBY == 1]
    
    return op_table_assmt['COLLECT_ID']

def return_img_list(args: argparse.Namespace) -> List[str]:
    """Retrieves image filenames from the provided directory."""
    if not args.img_directory:
        raise ValueError("The --image_list_filter flag requires --img_directory to be provided.")
        
    img_dir = args.img_directory
    print(f"Searching for images in: {img_dir}")
    
    # Use a more explicit list of extensions
    extensions = ['*.jpg', '*.tif', '*.png']
    image_path_list = []
    
    for ext in extensions:
        image_path_list.extend(glob.glob(str(img_dir / ext)))
    
    if not image_path_list:
        raise ValueError(f"No images found in directory: {img_dir}")  
    if args.tiled:
        tilenames = [os.path.basename(p) for p in image_path_list]
        filenames = list(map(lambda x: Utils.convert_tile_img_pth_to_basename(x), image_path_list))
    else:
        # Extract only the filenames (base names) for filtering against metadata's 'Filename' column
        tilenames = None
        filenames = [os.path.basename(p) for p in image_path_list]

    return sorted(filenames), sorted(tilenames)

def usable_processed_metadata(args: argparse.Namespace) -> pd.DataFrame:
    """
    Filters and cleans metadata based on usability flags and user-defined filters.
    """
    
    meta_df = load_metadata(args)
    print(f"Original metadata shape: {meta_df.shape}")
    
    # 1. Apply Usability Filters (Assuming columns exist as per documentation)
    # Filter 1: Metadata Thresholds
    meta_potentially_usable = meta_df[meta_df['Metadata Thresholds'] == "Pass"].copy()
    
    # Filter 2: Usability Random Forest (RF) and Final Usability
    meta_usable_rf = meta_potentially_usable[meta_potentially_usable['Usability Random Forest'] == "Pass"].copy()
    meta_usable = meta_df[meta_df['Usability'] == "Usable"].copy() # Use the final Usability column
    
    print(f"Metadata passing 'Usability' flag: {meta_usable.shape[0]}")
    if not meta_potentially_usable.empty:
        # Use .shape[0] for row count
        rf_portion = meta_usable_rf.shape[0] / meta_potentially_usable.shape[0]
        print(f"Portion usable (RF Pass / Threshold Pass): {rf_portion:.4f}")
    else:
        print("Warning: No images passed the 'Metadata Thresholds'.")

    # 2. Apply User Filter
    if args.goby_collects_filter:
        collect_list = filter_goby_collects(args.op_table_pth)
        meta_usable_filtered = meta_usable[meta_usable.collect_id.isin(collect_list)].copy()
        filter_type = "GOBY Collects"
    
    elif args.collect_id:
        collect_list = [args.collect_id]
        meta_usable_filtered = meta_usable[meta_usable.collect_id.isin(collect_list)].copy()
        filter_type = f"Collect ID: {args.collect_id}"

    elif args.image_list_filter:
        image_list, tile_list = return_img_list(args)
        if args.tiled:
            assert len(image_list) == len(tile_list)
            df = pd.DataFrame(np.c_[image_list, tile_list], columns=["Filename", "Tilename"])
            meta_usable_filtered = meta_usable[meta_usable.Filename.isin(image_list)].copy()
            meta_usable_filtered = pd.merge(df, meta_usable_filtered, on="Filename", how="left")
            meta_usable_filtered = meta_usable_filtered.rename(columns={"Filename":"BaseFilename"})
            meta_usable_filtered = meta_usable_filtered.rename(columns={"Tilename":"Filename"})
        else:
            meta_usable_filtered = meta_usable[meta_usable.Filename.isin(image_list)].copy()
        filter_type = "Image List"
    
    else:
        # This shouldn't be reached due to required=True in argparse group
        print("Error: No filtering option was selected. Exiting.")
        sys.exit(1)

    filt_shape = meta_usable_filtered.shape[0]
    print(f"Metadata filtered by {filter_type} count: {filt_shape}")
    
    # 3. Final Cleaning and Consistency Checks
    
    # Check 1: Duplicates/Missing in primary ID columns
    
    # Filename Deduplication Check
    drop_fnd = meta_usable_filtered.drop_duplicates(subset="Filename").shape[0]
    assert filt_shape - drop_fnd == 0, f"ERROR: {filt_shape - drop_fnd} duplicate 'Filename' entries found."
    
    # Filename NaN Check
    drop_fnn = meta_usable_filtered.dropna(subset=["Filename"]).shape[0]
    assert filt_shape - drop_fnn == 0, f"ERROR: {filt_shape - drop_fnn} missing 'Filename' entries found."
    
    # Image Path Deduplication Check
    # NOTE: The metadata must be deduplicated by 'Filename' or 'image_path' prior to this script 
    # if a single file appears multiple times. We only check for duplicates here.
    drop_ipd = meta_usable_filtered.drop_duplicates(subset=["image_path"]).shape[0]
    if args.tiled:
        assert drop_ipd - len(list(set(image_list))) == 0 , f"ERROR {drop_ipd - len(list(set(image_list)))} duplicate 'image_path' entries found."
    else:
        assert filt_shape - drop_ipd == 0, f"ERROR: {filt_shape - drop_ipd} duplicate 'image_path' entries found."
    
    # Time_s NaN Check
    drop_tsn = meta_usable_filtered.dropna(subset=["Time_s"]).shape[0]
    assert filt_shape - drop_tsn == 0, f"ERROR: {filt_shape - drop_tsn} missing 'Time_s' (timestamps) entries found."
    
    # 4. Altitude (DistanceToBottom_m) Check and Final Drop
    drop_dbn = meta_usable_filtered.dropna(subset=["DistanceToBottom_m"]).shape[0]
    missing_alt = filt_shape - drop_dbn
    
    if missing_alt > 0:
        print(f"Warning: {missing_alt} records missing 'DistanceToBottom_m'. These will be dropped.")
        # Dropping nan in DistanceToBottom_m to get final table
        meta_usable_filtered = meta_usable_filtered.dropna(subset=["DistanceToBottom_m"])

    # Final check for collect_id existence
    if 'collect_id' not in meta_usable_filtered.columns:
        print("Error: 'collect_id' column not found in metadata DataFrame after filtering. Exiting.")
        sys.exit(1)
        
    print(f"Final processed table shape: {meta_usable_filtered.shape}")
    
    # Calculate median time interval for QA
    medians = meta_usable_filtered.groupby('collect_id')['Time_s'].apply(lambda x: x.diff()).values
    print(f"Average median time interval (for QA): {np.nanmedian(medians):.4f} seconds.")
    
    return meta_usable_filtered

# --- Main Execution ---
def main():
    args = parse_args()
    
    # 1. Generate timestamp for output filename
    t = datetime.now()
    Ymmdd = f"{t.year:04d}{t.month:02d}{t.day:02d}"
    print(f"YYYYmmdd: {Ymmdd}")
    
    # 2. Process metadata
    processed_metadata = usable_processed_metadata(args)
    
    # 3. Determine output path and save
    if args.collect_id:
        filename = f"processed_metadata_{args.collect_id}_{Ymmdd}.csv"
    elif args.image_list_filter:
        # Use image_list_filter in filename for clarity
        filename = f"processed_metadata_image_list_filter_{Ymmdd}.csv" 
    else: # Default to goby_collects_filter
        filename = f"processed_metadata_goby_collects_filter_{Ymmdd}.csv"

    # CRITICAL BUG FIX: args.out_folder -> args.output_folder
    path = args.output_folder / filename

    # Ensure output directory exists
    path.parent.mkdir(parents=True, exist_ok=True)
    
    processed_metadata.to_csv(path, index=False)
    print(f"Processing complete. Final DataFrame shape: {processed_metadata.shape}")
    print(f"Saved to: {path}")

if __name__ == "__main__":
    main()