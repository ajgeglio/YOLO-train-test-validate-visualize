import argparse
import os
import sys
import pathlib
import json
from tqdm import tqdm

# Add src directory to path
SCRIPT_DIR = pathlib.Path(__file__).parent if '__file__' in locals() else pathlib.Path.cwd()
sys.path.append(str(SCRIPT_DIR.parent.parent / "src"))
from utils import Utils


def parse_args():
    parser = argparse.ArgumentParser(description="Convert Darwin JSON annotations to YOLO label text files.")
    parser.add_argument("--directory", type=str, required=True, help="Directory containing 'json' and 'labels' subfolders.")
    return parser.parse_args()


def main():
    args = parse_args()
    base_dir = pathlib.Path(args.directory)

    json_dir = base_dir / "json"
    labels_dir = base_dir / "labels"

    # Validate directories
    if not json_dir.exists():
        raise FileNotFoundError(f"Missing folder: {json_dir}")
    if not labels_dir.exists():
        print(f"Creating labels directory: {labels_dir}")
        labels_dir.mkdir(parents=True, exist_ok=True)

    json_files = list(json_dir.glob("*.json"))
    if not json_files:
        print(f"No JSON files found in {json_dir}")
        return

    print(f"Found {len(json_files)} JSON files to convert")

    for json_file in tqdm(json_files, desc="Converting", unit="file"):
        # Convert using your existing utility
        Utils.darwin_to_YOLO(str(json_file))

        # Move the generated .txt file into labels/
        txt_name = json_file.stem + ".txt"
        src_txt = json_file.with_suffix(".txt")
        dst_txt = labels_dir / txt_name

        if src_txt.exists():
            os.replace(src_txt, dst_txt)
        else:
            print(f"Warning: Expected output file not found: {src_txt}")

    print("Conversion complete.")


if __name__ == "__main__":
    main()