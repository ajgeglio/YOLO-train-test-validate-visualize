import pandas as pd
import numpy as np 
from ultralytics import YOLO
import argparse
from timeit import default_timer as stopwatch
import torch
import os
import sys

# Define a simple logging class to tee output to both console and a file
class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def parse_arguments():
    """Parse command-line arguments. Detection-specific args removed for clarity."""
    parser = argparse.ArgumentParser(description="Yolov8 classification validation statistics")
    parser.add_argument('--data_dir', dest='data_dir', default=None, help="point to directory of images to perform validation on")
    parser.add_argument('--weights', dest="weights", default=r'path\to\weights.pt', help='Trained weights path (should be a -cls model)')
    parser.add_argument('--split', dest="split", default='test', help='split to do val on yaml file: train, test, val')
    parser.add_argument('--output_name', dest='output_name', default="classification_output", type=str, help='name of the ouput folder created in the validation folder')
    parser.add_argument('--batch_size', dest='batch_size', default=4, type=int, help='Sets the number of images per batch.')
    # Removed: confidence, iou, save_hybrid (irrelevant for classification)
    return parser.parse_args()

if __name__ == '__main__':
    start_time = stopwatch()
    
    # --- Setup ---
    args = parse_arguments()
    folder = os.path.join("output", "validation", "classify", args.output_name)
    os.makedirs(folder, exist_ok=True)
    log_file_path = os.path.join(folder, f"{args.output_name}_log.txt")
    
    sys.stdout = Logger(log_file_path)

    print("--- Validation Arguments ---")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    print("----------------------------\n")

    torch.cuda.empty_cache()

    model = YOLO(args.weights)
    
    # Attempt to get imgsz from checkpoint, use default if needed
    try:
         image_size = model.ckpt["train_args"]["imgsz"]
    except:
         image_size = 224
         
    # --- Run Validation ---
    print("Starting YOLOv8 classification validation...")
    
    # Note: project=folder ensures all outputs go into the named folder
    metrics = model.val(
        data=args.data_dir,
        split=args.split,
        device=0,
        half=True,
        batch=args.batch_size,
        project=folder,
        plots=True, # Saves confusion_matrix.png automatically
        imgsz=image_size,
        save_json=True
    )
    
    print("\n--- Classification Validation Complete ---")

    # --- 1. Save Classification Report (CSV) ---
    print("\n--- Generating Classification Report ---")

    # Extract the confusion matrix and class names
    cm = metrics.confusion_matrix.matrix
    class_names = model.names
    
    # Calculate True Positives (TP), False Positives (FP), and False Negatives (FN)
    TP = np.diag(cm)
    FP = np.sum(cm, axis=0) - TP
    FN = np.sum(cm, axis=1) - TP

    # Calculate P, R, F1, and Support
    with np.errstate(divide='ignore', invalid='ignore'):
        precision = np.nan_to_num(TP / (TP + FP))
        recall = np.nan_to_num(TP / (TP + FN))
        f1_score = np.nan_to_num(2 * (precision * recall) / (precision + recall))

    support = np.sum(cm, axis=1)
    
    # Build the per-class report DataFrame
    # report_data = {
    #     'class': [class_names[i] for i in range(len(class_names))],
    #     'precision': precision,
    #     'recall': recall,
    #     'f1-score': f1_score,
    #     'support': support
    # }
    # report_df = pd.DataFrame(report_data)

    # # Calculate and append summary metrics
    # valid_indices = support > 0
    # total_support = np.sum(support)

    # overall_rows = pd.DataFrame([
    #     {'class': 'macro avg', 
    #      'precision': np.mean(precision[valid_indices]), 
    #      'recall': np.mean(recall[valid_indices]), 
    #      'f1-score': np.mean(f1_score[valid_indices]), 
    #      'support': total_support},
    #     {'class': 'weighted avg', 
    #      'precision': np.sum(precision * support) / total_support, 
    #      'recall': np.sum(recall * support) / total_support, 
    #      'f1-score': np.sum(f1_score * support) / total_support, 
    #      'support': total_support}
    # ])
    # report_df = pd.concat([report_df, overall_rows], ignore_index=True)
    
    # # Save the CSV
    # csv_path = os.path.join(folder, 'classification_report.csv')
    # report_df.to_csv(csv_path, index=False, float_format='%.4f')
    # print(f"Classification Report (CSV) saved to: {csv_path}")
    
    # # --- 2. Confirm Confusion Matrix (PNG) Save Location ---
    # # The plot is saved automatically by plots=True in val()
    # cm_plot_path = os.path.join(folder, 'confusion_matrix.png')
    # print(f"Confusion Matrix (PNG) saved to: {cm_plot_path}")
    # print("--------------------------------------\n")
    
    # # --- Save Overall Top-1/Top-5 Metrics to a separate CSV for easy summary ---
    # output_csv = os.path.join(folder, f"{args.output_name}_summary_metrics.csv")
    # df_metrics = pd.DataFrame(
    #     [[metrics.top1, metrics.top5]], 
    #     columns=["top1_accuracy", "top5_accuracy"]
    # )
    # df_metrics.to_csv(output_csv, index=False)
    
    end_time = stopwatch()
    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")