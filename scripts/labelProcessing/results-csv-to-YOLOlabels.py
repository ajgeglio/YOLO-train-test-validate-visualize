import sys
import glob
import argparse
import pathlib
SCRIPT_DIR = pathlib.Path(__file__).parent if '__file__' in locals() else pathlib.Path.cwd()
sys.path.append(str(SCRIPT_DIR.parent.parent / "src"))
from utils import Utils
import pandas as pd
import os
from tqdm import tqdm

def parse_arguments():
    """Parses command-line arguments for the tiling script."""
    parser = argparse.ArgumentParser(description="Converting inference output to YOLO labels")
    parser.add_argument('--results_file', required=True, help="Inference results file generated with runResults.py which was generated with a set confidence threshold.")
    return parser.parse_args()

def convert_predictions_to_yolo(args):
    """
    Convert results CSV to YOLO format labels.
    This function reads the output of YOLO inference: inference_results***.csv, processes the data,
    and saves the predictions as labels in YOLO format in a specified directory - used for sending to innodata.

    The img_list_file is optional and can be used to ensure that all images in the list have a corresponding label file.
    
    Args:
        predictions_file (str): Path to the CSV file containing predictions.
    """
    # Read the CSV file
    results_file = args.results_file
    df = pd.read_csv(results_file, index_col=0)
    output_dir = os.path.dirname(results_file)
    save_folder = pathlib.Path(output_dir) / "yolo_labels"
    # Ensure the output directory exists
    os.makedirs(save_folder, exist_ok=True)

    # Check if the necessary columns are present
    required_columns = ['Filename', 'cls', 'x', 'y', 'w', 'h']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"CSV file must contain the following columns: {', '.join(required_columns)}")
    # Determine the directory for the output labels
    output_dir = pathlib.Path(output_dir) / "yolo_labels"

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Function to process each image group and save the YOLO label file
    def save_yolo_labels(group):
        """
        Formats the grouped DataFrame into YOLO label strings and saves them to a .txt file.
        The YOLO format is: <class_id> <x_center> <y_center> <width> <height> (all normalized 0-1)
        """
        # Select 'cls' and the normalized box coordinates
        yolo_data = group[['cls', 'x', 'y', 'w', 'h']].values
        
        # Format each row as a space-separated string
        lines = []
        for row in yolo_data:
            if row[0] is None or pd.isna(row[0]):
                continue  # Skip empty rows
            # Ensure the class ID is an integer and the coordinates are floats
            # Format the floating point numbers to a suitable precision (e.g., 6 decimal places)
            lines.append(f"{int(row[0])} {row[1]:.6f} {row[2]:.6f} {row[3]:.6f} {row[4]:.6f}")
            
        # Get the image filename and strip the extension for the label file name
        image_filename = group['Filename'].iloc[0]
        base_filename, _ = os.path.splitext(image_filename)
        
        # Define the output path
        label_path = os.path.join(output_dir, f"{base_filename}.txt")
        
        # Write the labels to the file
        with open(label_path, 'w') as f:
            f.write('\n'.join(lines))

    # Group by Filename and apply the function to create label files
    groups = df.groupby("Filename")

    for name, group in tqdm(groups, total=len(groups)):
        save_yolo_labels(group)

    # save a labels.txt file with the full file paths
    labels_txt_path = os.path.join(os.path.dirname(output_dir), "labels.txt")
    with open(labels_txt_path, 'w') as f:
        for label_file in os.listdir(output_dir):
            if label_file.endswith(".txt"):
                f.write(os.path.abspath(os.path.join(output_dir, label_file)) + "\n")

    print(f"Successfully generated {len(os.listdir(output_dir))} YOLO label files in the '{output_dir}' directory.")

if __name__ == "__main__":
    args = parse_arguments()
    convert_predictions_to_yolo(args)
