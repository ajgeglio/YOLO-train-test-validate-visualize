import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from results import YOLOResults
import argparse
import time
import subprocess

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Yolov8 inference, prediction and scoring for goby detection")
    parser.add_argument('--has_labels', action="store_true", help='Argument to do inference and compare with labels')
    parser.add_argument('--has_cages', action="store_true", help='Argument to calculate fish intersection with quadrats')
    parser.add_argument('--img_directory', default=None, help='Directory of Images')
    parser.add_argument('--img_list_csv', default=None, help='Path to csv list of image paths')
    parser.add_argument('--lbl_list_csv', default=None, help='Path to csv list of label paths')
    parser.add_argument('--weights', default=r"src\models\GobyFinderAUV.pt", help='Weights path')
    parser.add_argument('--metadata', default=None, help='path to image metadata')
    parser.add_argument('--substrate', default=None, help='path to substrate output')
    parser.add_argument('--op_table', default=None, help = "path to operations level database table")
    parser.add_argument('--start_batch', default=0, type=int, help='Start at batch if interrupted')
    parser.add_argument('--plot', action="store_true", help='Argument to plot label + prediction overlay images')
    parser.add_argument('--supress_log', action="store_true", help='Suppress local terminal log')
    parser.add_argument('--output_name', default="inference_output", type=str, help='Name of the output csv')
    parser.add_argument('--batch_size', default=4, type=int, help='Batch size of n images in the inference loop')
    parser.add_argument('--img_size', default=2048, type=int, help='Max image dimension')
    parser.add_argument('--iou', default=0.6, type=float, help='IoU threshold for Non-Maximum Suppression')
    parser.add_argument('--confidence', default=0.01, type=float, help='Minimum confidence to call a detection')
    parser.add_argument('--results_confidence', default=0.2, type=float, help='Minimum confidence in results output table')
    parser.add_argument('--verify', action="store_true", help='Verify image before processing')
    return parser.parse_args()

def run_predict(img_directory, image_list_csv, weights, output_name, batch_size, start_batch, conf_thresh, has_labels, overlays):
    """Run the YOLO prediction script with the specified parameters."""
    cmd = [
        sys.executable,  # Use the current Python interpreter
        os.path.join(os.path.dirname(__file__), "batchpredict.py"),
        "--img_directory", img_directory,
        "--img_list_csv", image_list_csv,
        "--weights", weights,
        "--output_name", output_name,
        "--batch_size", str(batch_size),
        "--confidence", str(conf_thresh),
        "--start_batch", str(start_batch)
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
    return yolores

def main():
    args = parse_arguments()
    # Ensure the output directory exists
    output_name = args.output_name  
    run_path = os.path.join("output", "test_runs" if args.has_labels else "inference", output_name)

    # Run the prediction
    run_predict(
        img_directory=args.img_directory,
        weights=args.weights,
        output_name=output_name,
        batch_size=args.batch_size,
        has_labels=args.has_labels,
        overlays=args.plot,
        conf_thresh=args.confidence,
        start_batch=args.start_batch
    )

    # Process results
    yolo_infer_path = os.path.join(run_path, "predictions.csv")
    meta_path = args.metadata  # Assuming this is the metadata path
    substrate_path = args.substrate  # Assuming substrate_path is not used in this context
    op_path = args.op_table  # Assuming op_path is not used in this context
    results_confidence = args.results_confidence

    yolores = output_YOLO_results(meta_path, yolo_infer_path, substrate_path, op_path, results_confidence, find_closest=True)
    # Save or further process yolores as needed
    out_path = os.path.join(run_path, f"{args.output_name}_results.csv")
    print("saving YOLO results to csv", out_path)
    yolores.to_csv(out_path, index=False)
    print(yolores.head())

if __name__ == "__main__":
    main()