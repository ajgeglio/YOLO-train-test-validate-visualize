import subprocess
import sys
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Run YOLO batch prediction and results script.")
    parser.add_argument("--img_directory", required=False, help="Path to image directory")
    parser.add_argument("--img_list_csv", required=False, help="Path to image list csv")
    parser.add_argument("--weights", required=False, default=r"path\to\model_weights.pt", help="Path to YOLO model weights")
    parser.add_argument("--op_table", default=r"Z:\__AdvancedTechnologyBackup\07_Database\OP_TABLE.xlsx", help="Path to OP_TABLE.xlsx")
    parser.add_argument("--metadata", required=True, help="Path to meta.csv")
    parser.add_argument("--output_name", required=True, help="Output name for results")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--confidence", type=float, default=0.01, help="Confidence threshold")
    parser.add_argument("--results_confidence", type=float, default=0.2, help="Results confidence threshold")
    parser.add_argument("--start_batch", type=int, default=0, help="Start batch index")
    parser.add_argument("--has_labels", action="store_true", help="Whether labels are present")
    parser.add_argument("--run_predict", action="store_true", help="Whether to run prediction")
    parser.add_argument("--plot", action="store_true", help="Whether to plot overlays")
    parser.add_argument("--substrate_path", help="Optional substrate path") 
    parser.add_argument('--use_img_size', action='store_true', help="perform inference on images without defaulting to the weights default")
    parser.add_argument('--resume', action='store_true', help='use predictions.py file to continue')
    args = parser.parse_args()

    cmd = [
        sys.executable,
        os.path.join(os.path.dirname(__file__), "batchPredict+results.py"),
        "--weights", args.weights,
        "--output_name", args.output_name,
        "--batch_size", str(args.batch_size),
        "--confidence", str(args.confidence),
        "--metadata", args.metadata,
        "--op_table", args.op_table,
        "--results_confidence", str(args.results_confidence),
        "--start_batch", str(args.start_batch),
    ]
    # Conditionally add the directory OR the CSV path
    if args.img_directory is not None:
        cmd.extend(["--img_directory", args.img_directory])
    elif args.img_list_csv is not None:
        cmd.extend(["--img_list_csv", args.img_list_csv])
    else:
        cmd.extend(["--img_directory", "default_directory"])  # Provide a default or handle this case appropriately
    if args.use_img_size:
        cmd.extend(["--use_img_size"])
    if args.resume:
        cmd.extend(["--resume"])
    if args.run_predict:
        cmd.append("--run_predict")
    if args.has_labels:
        cmd.append("--has_labels")
    if args.plot:
        cmd.append("--plot")
    if args.substrate_path:
        cmd.extend(["--substrate_path", args.substrate_path])

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()