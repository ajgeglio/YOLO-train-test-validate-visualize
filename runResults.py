import subprocess
import sys
import os

# Example usage
if __name__ == "__main__":
    # Define paths and parameters for the YOLO prediction script
    # These paths should be adjusted according to your Experiment setup
    img_directory = r"Z:\__Organized_Directories_InProgress\GobyFinderDatasets\AUV_datasets\test_sets\challenge_test\images"
    op_path = r"Z:\__AdvancedTechnologyBackup\01_DerivedProducts\Database\OP_TABLE.xlsx"
    substrate_path = None  # Assuming substrate_path is not used in this context
    meta_path = r"Z:\__Organized_Directories_InProgress\GobyFinderDatasets\AUV_datasets\test_sets\challenge_test\meta.csv"
    output_name = "challenge_test_test_run"
    has_labels = True
    overlays = True
    batch_size = 4
    confidence = 0.01
    results_confidence = 0.2
    start_batch = 0
    
    cmd = [
            sys.executable,  # Use the current Python interpreter
            os.path.join(os.path.dirname(__file__), "batchPredict+results.py"),
            "--img_directory", img_directory,
            "--output_name", output_name,
            "--batch_size", str(batch_size),
            "--confidence", str(confidence),
            "--metadata", meta_path,
            "--op_table", op_path,
            "--results_confidence", str(results_confidence),
            "--start_batch", str(start_batch)
        ]
    if has_labels:
        cmd.append("--has_labels")
    if overlays:
        cmd.append("--plot")
    # You can add more arguments as needed

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)