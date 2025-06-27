import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from results import YOLOResults
import subprocess

def run_predict(img_directory, weights, output_name, batch_size, has_labels=False, overlays=False, conf_thresh=0.001):
    """Run the YOLO prediction script with the specified parameters."""
    cmd = [
        sys.executable,  # Use the current Python interpreter
        os.path.join(os.path.dirname(__file__), "batchpredict.py"),
        "--img_directory", img_directory,
        "--weights", weights,
        "--output_name", output_name,
        "--batch_size", str(batch_size),
        "--confidence", str(conf_thresh)
    ]
    if has_labels:
        cmd.append("--has_labels")
    if overlays:
        cmd.append("--plot")
    # You can add more arguments as needed

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

def output_YOLO_results(meta_path, yolo_infer_path, substrate_path, op_path, conf_thresh, find_closest=False):
    # Initialize the output class
    output = YOLOResults(meta_path, yolo_infer_path, substrate_path, op_path, conf_thresh)
    yolores = output.yolo_results(find_closest=find_closest)
    # Save or further process yolores as needed
    out_path = os.path.join("output", "test_runs", output_name, f"{output_name}_results.csv")
    print("saving YOLO results to csv", out_path)
    yolores.to_csv(out_path, index=False)
    return yolores

# Example usage
if __name__ == "__main__":
    # Define paths and parameters for the YOLO prediction script
    # These paths should be adjusted according to your Experiment setup
    img_directory = r"Z:\__Organized_Directories_InProgress\GobyFinderDatasets\AUV_datasets\test_sets\challenge_test\images"
    weights = r"Z:\__AdvancedTechnologyBackup\04_ProjectData\Proj_GobyFinder\gobyfinder_yolov8\Best_Model_Weights\train\run12+_bbox_png\train\weights\best.pt"
    op_path = r"Z:\__AdvancedTechnologyBackup\01_DerivedProducts\Database\OP_TABLE.xlsx"
    substrate_path = None  # Assuming substrate_path is not used in this context
    output_name = "challenge_test_test_run"
    yolo_infer_path = os.path.join("output", "test_runs", output_name, f"{output_name}_predictions.csv")
    meta_path = r"Z:\__Organized_Directories_InProgress\GobyFinderDatasets\AUV_datasets\test_sets\challenge_test\meta.csv"  # Assuming meta_path is not used in this context
    
    # Parameters for the YOLO prediction script
    batch_size = 1
    has_labels = True
    conf_thresh = 0.3
    overlays = True  # Set to True if you want to generate overlays


    print("Running YOLOResults output script...")
    # run_predict(img_directory, weights, output_name, batch_size, has_labels, conf_thresh=0.001, overlays=overlays)
    yolores = output_YOLO_results(meta_path, yolo_infer_path, substrate_path, op_path, conf_thresh, find_closest=True)
    print(yolores.head())
