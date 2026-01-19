import sys
import os
import argparse
import shutil
import glob
from datetime import datetime
import pathlib
SCRIPT_DIR = pathlib.Path(__file__).parent if '__file__' in locals() else pathlib.Path.cwd()
sys.path.append(str(SCRIPT_DIR.parent.parent / "src"))
from AUVMetadataProcessor import AUVDataProcessor


def parse_args():
    parser = argparse.ArgumentParser(description="Run AUVDataProcessor for specified years and directories.")
    parser.add_argument("--years", type=str, required=True, help="Comma-separated list of years, e.g. '2019,2020,2021'.")
    parser.add_argument(
        "--output_directory", 
        type=str, 
        default=r"Z:\__AdvancedTechnologyBackup\07_Database\MetadataCombined", 
        help="directory to output updated master metadata"
    )
    parser.add_argument(
        "--unpacked_image_directory", 
        type=str, 
        default=r"Z:\__Organized_Directories_InProgress",
        help="Directory containing unpacked images."
    )
    parser.add_argument(
        "--metadata_directory",
        type=str,
        default=r"Z:\__AdvancedTechnologyBackup\01_DerivedProducts\CollectMetadata",
        help="Directory containing metadata files."
    )
    return parser.parse_args()

def make_run_id(years):
    # Convert to integers for safety, then back to strings
    years_int = [int(y) for y in years]
    earliest = min(years_int)
    latest = max(years_int)

    today = datetime.today().strftime("%Y%m%d")

    return f"{earliest}-{latest}-{today}"

def main():
    args = parse_args()
    # Convert comma-separated years into a list
    years = [y.strip() for y in args.years.split(",")]
    print("Updating metadata for", years)

    run_id = make_run_id(years)
    out_folder = os.path.join(args.output_directory, run_id)
    os.makedirs(out_folder, exist_ok=True)

    # Initialize processor
    processor = AUVDataProcessor(
        out_folder=out_folder,
        unpacked_image_directory=args.unpacked_image_directory,
        metadata_directory=args.metadata_directory,
        years=years
    )

    # Update collect lists
    collects = processor.update_collect_lists(years)
    print("Total number of collects", len(collects))

    # Update unpacked images metadata
    processor.update_unpacked_images_metadata(years=years)


if __name__ == "__main__":
    main()