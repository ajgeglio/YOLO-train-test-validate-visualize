import glob
import os
import shutil
import zipfile
import zlib
import logging
from pathlib import Path
from typing import List, Tuple
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

def argument_parser():

    parser = argparse.ArgumentParser(description="Zip folders, verify, and delete originals.")
    parser.add_argument(
        "--folders",
        type=str,
        nargs="+",
        required=False,
        help="List of folder paths or glob patterns to process.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="If set, do not create zips or delete folders; just simulate.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="If set, log each file being processed.",
    )
    return parser.parse_args()

def compute_crc32(path: Path, chunk_size: int = 65536) -> int:
    """Compute CRC32 for a file, returning unsigned 32-bit int."""
    crc = 0
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            crc = zlib.crc32(chunk, crc)
    return crc & 0xFFFFFFFF

def collect_files(folder: Path) -> List[Path]:
    """Return list of file paths under folder, relative to folder root, sorted."""
    files = [p for p in folder.rglob("*") if p.is_file()]
    files.sort()
    return files

def create_zip_from_folder(folder: Path, zip_path: Path, verbose: bool = False) -> None:
    """Create zip archive preserving relative paths."""
    # Use zipfile to preserve CRC info
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in collect_files(folder):
            arcname = str(file_path.relative_to(folder))
            if verbose:
                logging.info("Adding %s as %s", file_path, arcname)
            zf.write(file_path, arcname=arcname)

def verify_zip_matches_folder(folder: Path, zip_path: Path, verbose: bool = False) -> Tuple[bool, str]:
    """Verify that zip contains same files and matching CRC32 checksums."""
    if not zip_path.exists():
        return False, f"Zip file {zip_path} does not exist."

    folder_files = collect_files(folder)
    folder_rel = [str(p.relative_to(folder)).replace("\\", "/") for p in folder_files]

    with zipfile.ZipFile(zip_path, "r") as zf:
        zip_infos = {zi.filename: zi for zi in zf.infolist() if not zi.is_dir()}
        zip_files = sorted(zip_infos.keys())

        # Compare file lists
        if sorted(folder_rel) != zip_files:
            missing_in_zip = set(folder_rel) - set(zip_files)
            extra_in_zip = set(zip_files) - set(folder_rel)
            msg = []
            if missing_in_zip:
                msg.append(f"Missing in zip: {sorted(missing_in_zip)}")
            if extra_in_zip:
                msg.append(f"Extra in zip: {sorted(extra_in_zip)}")
            return False, "File list mismatch. " + " ".join(msg)

        # Compare CRC32 for each file
        for file_path in folder_files:
            rel = str(file_path.relative_to(folder)).replace("\\", "/")
            expected_crc = compute_crc32(file_path)
            zi = zip_infos.get(rel)
            if zi is None:
                return False, f"Entry {rel} missing in zip."
            zip_crc = zi.CRC
            if expected_crc != zip_crc:
                return False, f"CRC mismatch for {rel}: folder={expected_crc} zip={zip_crc}"

    return True, "Verification succeeded"

def safe_remove_folder(folder: Path, verbose: bool = False) -> None:
    """Remove folder tree after verification."""
    if verbose:
        logging.info("Removing folder %s", folder)
    shutil.rmtree(folder)

def process_batch_folders(folders: List[str], dry_run: bool = True, verbose: bool = False) -> dict:
    """Process each folder: zip, verify, and delete original on success."""
    summary = {"processed": 0, "zipped": 0, "verified": 0, "deleted": 0, "errors": []}

    for folder_glob in folders:
        # Expand glob if user passed patterns; if already concrete path, keep it
        matched = glob.glob(folder_glob) if any(ch in folder_glob for ch in ["*", "?","["]) else [folder_glob]
        for folder_str in matched:
            folder = Path(folder_str)
            summary["processed"] += 1
            if not folder.exists() or not folder.is_dir():
                msg = f"Folder not found or not a directory: {folder}"
                logging.error(msg)
                summary["errors"].append(msg)
                continue

            parent = folder.parent
            zip_name = folder.name.rstrip("/\\") + ".zip"
            zip_path = parent / zip_name

            try:
                logging.info("Processing folder %s", folder)
                if dry_run:
                    logging.info("[dry run] Would create zip at %s", zip_path)
                else:
                    create_zip_from_folder(folder, zip_path, verbose=verbose)
                    summary["zipped"] += 1

                # Verify
                if dry_run:
                    logging.info("[dry run] Would verify zip %s", zip_path)
                    verified = True
                    verify_msg = "Dry run assumed verification success"
                else:
                    verified, verify_msg = verify_zip_matches_folder(folder, zip_path, verbose=verbose)

                if verified:
                    logging.info("Verified %s: %s", zip_path, verify_msg)
                    summary["verified"] += 1
                    if dry_run:
                        logging.info("[dry run] Would delete folder %s", folder)
                    else:
                        safe_remove_folder(folder, verbose=verbose)
                        summary["deleted"] += 1
                else:
                    msg = f"Verification failed for {zip_path}: {verify_msg}"
                    logging.error(msg)
                    summary["errors"].append(msg)
                    # Optionally remove incomplete zip to avoid confusion
                    # zip_path.unlink(missing_ok=True)  # Python 3.8+
            except Exception as e:
                msg = f"Error processing {folder}: {e}"
                logging.exception(msg)
                summary["errors"].append(msg)

    return summary

if __name__ == "__main__":
    args = argument_parser()
    folders = ['Z:\\__Organized_Directories_InProgress\\GobyFinderDatasets\\AUV_datasets\\innodata_2025\\2024 Transects Update\\tiled\\json',
                'Z:\\__Organized_Directories_InProgress\\GobyFinderDatasets\\AUV_datasets\\innodata_2025\\2024 Transects Update\\tiled\\labels',
                'Z:\\__Organized_Directories_InProgress\\GobyFinderDatasets\\AUV_datasets\\innodata_2025\\2024 Transects Update\\untiled\\images',
                'Z:\\__Organized_Directories_InProgress\\GobyFinderDatasets\\AUV_datasets\\innodata_2025\\2024 Transects Update\\untiled\\labels',
                'Z:\\__Organized_Directories_InProgress\\GobyFinderDatasets\\AUV_datasets\\innodata_2025\\2025 Raw Remus Update\\tiled\\json',
                'Z:\\__Organized_Directories_InProgress\\GobyFinderDatasets\\AUV_datasets\\innodata_2025\\2025 Raw Remus Update\\tiled\\labels',
                'Z:\\__Organized_Directories_InProgress\\GobyFinderDatasets\\AUV_datasets\\innodata_2025\\2025 Raw Remus Update\\untiled\\images',
                'Z:\\__Organized_Directories_InProgress\\GobyFinderDatasets\\AUV_datasets\\innodata_2025\\2025 Raw Remus Update\\untiled\\labels',
                'Z:\\__Organized_Directories_InProgress\\GobyFinderDatasets\\AUV_datasets\\innodata_2025\\HNM Relabel Update\\tiled\\json',
                'Z:\\__Organized_Directories_InProgress\\GobyFinderDatasets\\AUV_datasets\\innodata_2025\\HNM Relabel Update\\tiled\\labels',
                'Z:\\__Organized_Directories_InProgress\\GobyFinderDatasets\\AUV_datasets\\innodata_2025\\HNM Relabel Update\\untiled\\labels',
                'Z:\\__Organized_Directories_InProgress\\GobyFinderDatasets\\AUV_datasets\\innodata_2025\\Random Relabel Update\\tiled\\json',
                'Z:\\__Organized_Directories_InProgress\\GobyFinderDatasets\\AUV_datasets\\innodata_2025\\Random Relabel Update\\tiled\\labels',
                'Z:\\__Organized_Directories_InProgress\\GobyFinderDatasets\\AUV_datasets\\innodata_2025\\Random Relabel Update\\untiled\\labels']

    # Settings
    DRY_RUN = args.dry_run   # Set to False to actually create zips and delete folders
    VERBOSE = False  # Set to True for per-file logging

    result = process_batch_folders(folders, dry_run=DRY_RUN, verbose=VERBOSE)
    logging.info("Summary: %s", result)