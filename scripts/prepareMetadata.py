import argparse
import pathlib
from datetime import datetime
import pandas as pd
import numpy as np
import glob
import sys
import os

def parse_args():
    """
    Parses command-line arguments for the metadata analysis script.
    """
    parser = argparse.ArgumentParser(description="Filter and clean all image metadata to get usable GOBY-assessed collects for a specified year.")
    parser.add_argument("--metadata_folder", type=pathlib.Path, default=r"Z:\__AdvancedTechnologyBackup\07_Database\MetadataCombined",
        help="Path to the directory containing all_unpacked_images_metadata*.pkl/pickle files."
    )
    parser.add_argument("--out_folder", type=pathlib.Path, default=r"Z:\__AdvancedTechnologyBackup\07_Database\MetadataCombined",
        help="Path to the directory containing all_unpacked_images_metadata*.pkl/pickle files."
    )
    parser.add_argument("--op_table_pth", type=pathlib.Path, default=r"Z:\__AdvancedTechnologyBackup\07_Database\OP_TABLE.xlsx",
        help="Path to the operational assessment table (OP_TABLE.xlsx)."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--goby_collects_filter", action="store_true",
        help="Whether to apply filtering to the metadata"
    )
    group.add_argument("--collect_id", type=str,
        help="Specific collect_id to filter the metadata"
    )
    return parser.parse_args()

def filter_goby_collects(op_table_pth: pathlib.Path) -> pd.Series:
    """
    Reads the OP table and returns a Series of all COLLECT_IDs that are flagged 
    for GOBY assessments (GOBY == 1), without filtering by year.
    """
    try:
        op_table = pd.read_excel(op_table_pth)
    except FileNotFoundError:
        print(f"Error: OP table not found at {op_table_pth}")
        sys.exit(1)
        
    op_table_assmt = op_table[op_table.GOBY == 1]
    # Corrected logic to simply return the COLLECT_ID series from the GOBY-flagged rows.
    return op_table_assmt['COLLECT_ID']

def usable_processed_metadata(aluim: pd.DataFrame, collect_list: str) -> pd.DataFrame:
    """
    Filters and cleans metadata for GOBY-assessed collects that pass usability checks.
    
    Args:
        aluim: All unpacked images metadata DataFrame.
        op_table_pth: Path to the OP_TABLE.xlsx.
    
    Returns:
        A cleaned DataFrame of usable GOBY metadata.
    """
    
    meta_df = aluim
    print("original metadata", meta_df.shape)
    
    # NOTE: These columns ('Metadata Thresholds', 'Usability Random Forest', 'Usability')
    # must exist in the input 'aluim' DataFrame for this script to run correctly.
    meta_potentially_usable = meta_df[(meta_df['Metadata Thresholds'] == "Pass")]
    meta_usable_rf = meta_potentially_usable[(meta_potentially_usable['Usability Random Forest'] == "Pass")]
    meta_usable = meta_df[(meta_df['Usability'] == "Usable")]
    
    if not meta_potentially_usable.empty:
        # Use simple division for the check
        print("portion usable out of potential", meta_usable_rf.shape[0] / meta_potentially_usable.shape[0])
    else:
        print("Warning: No images passed the 'Metadata Thresholds'.")

    # Filter metadata to only include GOBY-flagged collects
    meta_usable_goby = meta_usable[meta_usable.collect_id.isin(collect_list)]
    
    # Checks and Cleaning
    orig_shape = meta_usable_goby.shape[0]
    print("Goby collects metadata", orig_shape)
    
    # Deduplication and NaN checks (printing counts of filtered rows)
    drop_fnd = meta_usable_goby.drop_duplicates(subset="Filename").shape[0]
    print(orig_shape - drop_fnd, "duplicate filenames")
    
    drop_fnn = meta_usable_goby.dropna(subset="Filename").shape[0]
    print(orig_shape - drop_fnn, "missing filenames")
    
    drop_ipd = meta_usable_goby.drop_duplicates(subset="image_path").shape[0]
    print(orig_shape - drop_ipd, "duplicate image paths")
    
    drop_tsn = meta_usable_goby.dropna(subset="Time_s").shape[0]
    print(orig_shape - drop_tsn, "missing timestamps")
    
    drop_dbn = meta_usable_goby.dropna(subset="DistanceToBottom_m").shape[0]
    print(orig_shape - drop_dbn, "missing altitudes")
    
    # Dropping nan in DistanceToBottom_m to get final table
    meta_usable_goby_dbn = meta_usable_goby.dropna(subset="DistanceToBottom_m")

    # ensure that the 'collect_id' or 'CollectID' column is present
    if 'collect_id' not in meta_usable_goby_dbn.columns and 'CollectID' not in meta_usable_goby_dbn.columns:
        print("Error: 'collect_id' or 'CollectID' column not found in metadata DataFrame.")
        sys.exit(1)
    print("final table", meta_usable_goby_dbn.shape)
    
    # Calculate median time interval
    medians = meta_usable_goby_dbn.groupby('collect_id')['Time_s'].apply(lambda x: x.diff()).values
    print("Average median time interval", np.nanmedian(medians))
    
    return meta_usable_goby_dbn

def main():
    args = parse_args()
    
    t = datetime.now()
    Ymmdd = f"{t.year:02d}{t.month:02d}{t.day:02d}"
    print("YYYYmmdd:", Ymmdd)
    
    # Check for metadata folder existence
    if not args.metadata_folder.exists():
        print(f"Error: metadata folder not found at {args.metadata_folder}, exiting.")
        sys.exit(1)

    print(f"Loading metadata from: {args.metadata_folder}")
    
    # Glob for files and sort to find the latest one
    # Note: If your files use '.pickle' extension, change '*.pkl' to '*.pickle' below
    metadata_files = list(args.metadata_folder.glob("all_unpacked_images_metadata*.pkl"))
    metadata_files.extend(list(args.metadata_folder.glob("all_unpacked_images_metadata*.pickle")))

    if not metadata_files:
        print(f"No metadata files found in {args.metadata_folder}, exiting.")
        sys.exit(1)

    # Sort files lexicographically (which correctly handles YYYYmmdd dates)
    metadata_files.sort(reverse=True)
    latest_metadata_file = metadata_files[0]
    
    print(f"Loading latest metadata file: {latest_metadata_file}")

    # Load the metadata
    try:
        metadata = pd.read_pickle(latest_metadata_file)
    except Exception as e:
        print(f"Error loading metadata pickle file: {e}")
        sys.exit(1)
    
    # Execute the core function
    # Removed the unused 'year' argument from the function call

    if args.goby_collects_filter:
        collect_list = filter_goby_collects(args.op_table_pth)
        print(len(collect_list), "all time goby collects")
        path = args.out_folder / f"processed_metadata_goby_collects_{Ymmdd}.csv"
    elif args.collect_id:
        collect_list = [args.collect_id]
        print(f"Filtering for collect_id: {collect_list}")

        if len(collect_list) == 1:
            path = args.out_folder / f"processed_metadata_{args.collect_id}_{Ymmdd}.csv"
        else:
            path = args.out_folder / f"processed_metadata_collect_filter_{Ymmdd}.csv"

    processed_metadata = usable_processed_metadata(
        aluim=metadata, 
        collect_list=collect_list
    )
    processed_metadata.to_csv(path, index=False)
    print(f"Processing complete. Final DataFrame shape: {processed_metadata.shape}")

if __name__ == "__main__":
    main()