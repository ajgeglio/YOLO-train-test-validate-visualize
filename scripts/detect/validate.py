import pandas as pd
import numpy as np 
from ultralytics import YOLO
import argparse
from timeit import default_timer as stopwatch
import torch
import os
import sys
"""
YOLOv8 Validation Module

This script performs validation on a trained YOLOv8 model using a dataset 
defined in a YAML file. It computes standard object detection metrics 
and exports results to CSV.

Attributes:
    metrics.box.p (list): Precision for each class at specified IoU.
    metrics.box.r (list): Recall for each class.
    metrics.box.f1 (list): F1 score for each class.
    metrics.box.all_ap (list): Average Precision (AP) for all classes across 10 IoU thresholds.
    metrics.box.map (float): Mean Average Precision (mAP) over all IoU thresholds.
    metrics.box.map50 (float): mAP at an IoU threshold of 0.50.

Methods:
    model.val(): Executes validation logic, returning a DetMetrics object.
    results_dict: Accesses a dictionary mapping metric keys to their computed values.
    curves_results: Retrieves data for plotting Precision-Recall and F1-Confidence curves.

Outputs:
    - {output_name}_metrics.csv: Summary of precision, recall, f1, and mAP.
    - {output_name}_curves.csv: Raw data for validation curves.
    - {output_name}_log.txt: Terminal output log.
"""
# Define a simple logging class to tee output to both console and a file
class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # This flush method is needed for compatibility with Python 3's sys.stdout
        self.terminal.flush()
        self.log.flush()

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Yolov8 validation statistics for goby detection, existing labels needed")
    parser.add_argument('--data_yaml', dest='data_yaml', default=None, help="yml file pointing to directory of images to perform inference on (must have labels)")
    parser.add_argument('--weights', dest="weights", default=r'path\to\weights.pt', help='Trained weights path')
    parser.add_argument('--split', dest="split", default='test', help='split to do val on yaml file: train, test, val')
    parser.add_argument('--output_name', dest='output_name', default="validation_output", type=str, help='name of the ouput csv, also the folder created in the validation folder')
    parser.add_argument('--batch_size', dest='batch_size', default=4, type=int, help='Sets the number of images per batch. Use -1 for AutoBatch, which automatically adjusts based on GPU memory availability')
    parser.add_argument('--confidence', dest='confidence', default=0.01, type=float, help='Sets the minimum confidence threshold for detections. Detections with confidence below this threshold are discarded.')
    parser.add_argument('--iou', dest='iou', default=0.6, type=float, help='Sets the Intersection Over Union (IoU) threshold for Non-Maximum Suppression (NMS). Helps in reducing duplicate detections.')
    return parser.parse_args()

def run_validation(args):    
    start_time = stopwatch()
    # --- Logging Setup ---
    folder = os.path.join("output", "validation", "detect", args.output_name)
    os.makedirs(folder, exist_ok=True)
    log_file_path = os.path.join(folder, f"{args.output_name}_log.txt")
    
    # Redirect standard output to both the console and the log file
    sys.stdout = Logger(log_file_path)

    # Log the command-line arguments
    print("--- Validation Arguments ---")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    print("----------------------------\n")

    torch.cuda.empty_cache()

    model = YOLO(args.weights)
    image_size = model.ckpt["train_args"]["imgsz"]
    # --- Run Validation ---
    print("Starting YOLOv8 validation...")
    metrics = model.val(
        data=args.data_yaml,
        split=args.split,
        device=0,  # Use GPU 0, change if needed
        verbose=True,
        half=True,
        batch=args.batch_size,
        project=folder,
        plots=True,
        conf=args.confidence,
        imgsz=image_size,
        iou=args.iou,
        save_json=True
    )
    
    print("\n--- Validation Complete ---")

    # --- Save Metrics to CSV ---
    output_csv = os.path.join(folder, f"{args.output_name}_metrics.csv")
    output_curves = os.path.join(folder, f"{args.output_name}_curves.csv")

    mAP = metrics.box.map
    map50 = metrics.box.ap50  # map50
    p = metrics.box.p
    r = metrics.box.r
    f1 = metrics.box.f1
    all_ap = metrics.box.all_ap

    ar = np.c_[p, r, f1, map50, mAP]
    pd.DataFrame(ar, columns=["precision", "recall", "f1", "mAP50", "mAP"]).to_csv(output_csv)
    pd.DataFrame(metrics.curves_results).to_csv(output_curves)

    end_time = stopwatch()
    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")


if __name__ == '__main__':
    args = parse_arguments()
    run_validation(args)