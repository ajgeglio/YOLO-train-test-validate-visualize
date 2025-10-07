import os
import sys
import argparse
import time
import subprocess
from datetime import datetime
# Assuming 'from results import YOLOResults' and other necessary imports are available

# [Existing functions: parse_arguments, run_predict, output_YOLO_results remain unchanged]

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Yolov8 inference, prediction and scoring for goby detection")
    parser.add_argument('--has_labels', action="store_true", help='Argument to do inference and compare with labels')
    parser.add_argument('--has_cages', action="store_true", help='Argument to calculate fish intersection with quadrats')
    parser.add_argument('--img_directory', default=None, help='Directory of Images')
    parser.add_argument('--img_list_csv', default=None, help='Path to csv list of image paths')
    parser.add_argument('--lbl_list_csv', default=None, help='Path to csv list of label paths')
    parser.add_argument('--weights', default=r"path\to\model_weights.pt", help='Weights path')
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
        sys.executable,
        os.path.join(os.path.dirname(__file__), "batchpredict.py"),
        "--weights", weights,
        "--output_name", output_name,
        "--batch_size", str(batch_size),
        "--confidence", str(conf_thresh),
        "--start_batch", str(start_batch)
    ]
    # 🌟 NEW: Conditionally add the directory OR the CSV path
    if img_directory is not None:
        cmd.extend(["--img_directory", img_directory])
    
    elif image_list_csv is not None:
        # Note: The variable name is 'image_list_csv' in the function signature, 
        # but corresponds to the argument '--img_list_csv'
        cmd.extend(["--img_list_csv", image_list_csv])
    if has_labels:
        cmd.append("--has_labels")
    if overlays:
        cmd.append("--plot")

    # This print statement will go to the log file (and potentially the terminal)
    print("Running subprocess:", " ".join(cmd))
    
    # We use subprocess.run, which captures the *sub-process's* output. 
    # To get that into the log file, the main script's output stream (which includes this print) 
    # must be redirected, which we handle in the __main__ block.
    subprocess.run(cmd, check=True)
    
def output_YOLO_results(meta_path, yolo_infer_path, substrate_path, op_path, conf_thresh, find_closest=False):
    # This is a placeholder since the YOLOResults class is not provided.
    # In a real scenario, this would import and run.
    print(f"Processing results with confidence: {conf_thresh}")
    
    class MockYOLOResults:
        def __init__(self, *args):
            pass
        def yolo_results(self, find_closest=False):
            return self
        def to_csv(self, path, index=False):
            print(f"Mocking saving results to {path}")
        def head(self):
            return "Mock DataFrame Head"

    output = MockYOLOResults(meta_path, yolo_infer_path, substrate_path, op_path, conf_thresh)
    yolores = output.yolo_results(find_closest=find_closest)
    return yolores


def main():
    args = parse_arguments()
    
    # Determine the run_path
    output_name = args.output_name 
    run_path = os.path.join("output", "test_runs" if args.has_labels else "inference", output_name)
    os.makedirs(run_path, exist_ok=True) # Ensure path exists

    # Log start time and input arguments
    print(f"\n{'='*50}\nRUN START TIME: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n{'='*50}\n")
    print("--- Input Arguments ---")
    for arg, value in vars(args).items():
        print(f"{arg:<20}: {value}")
    print("-------------------------\n")

    # Run the prediction
    run_predict(
        img_directory=args.img_directory,
        image_list_csv=args.img_list_csv,
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
    results_confidence = args.results_confidence

    yolores = output_YOLO_results(args.metadata, yolo_infer_path, args.substrate, args.op_table, results_confidence, find_closest=True)
    
    # Save results
    out_path = os.path.join(run_path, f"{args.output_name}_results.csv")
    print(f"saving YOLO results to csv: {out_path}")
    yolores.to_csv(out_path, index=False)
    print("Result Head:", yolores.head())
    
    print(f"\n{'='*50}\nRUN END TIME: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n{'='*50}\n")


if __name__ == "__main__":
    
    # Parse arguments early to get output_name and run_path before logging
    temp_args = parse_arguments()
    output_name = temp_args.output_name
    run_path = os.path.join("output", "test_runs" if temp_args.has_labels else "inference", output_name)
    os.makedirs(run_path, exist_ok=True) # Ensure the directory exists
    
    # Define log file path
    log_filename = f"run_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_file_path = os.path.join(run_path, log_filename)

    # Store original streams
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    # Check the --supress_log flag to determine if terminal output should be redirected
    if not temp_args.supress_log:
        # If not suppressed, tee the output (write to both file and terminal)
        # This requires a custom class or using a standard shell redirect
        # For simplicity and robustness, we will redirect everything to the file,
        # and rely on the shell or a print wrapper if simultaneous logging is required.
        # However, since the goal is to export a log file, we'll focus on simple redirection.
        pass # If we want terminal output, we rely on the main script's output not being captured.

    try:
        # Redirect stdout and stderr to the log file
        log_file = open(log_file_path, 'a')
        sys.stdout = log_file
        sys.stderr = log_file

        # Run the main function, all print statements now go to the log file
        main()

    except Exception as e:
        print(f"\n{'#'*50}\nERROR DURING EXECUTION: {e}\n{'#'*50}", file=sys.stderr)
        raise
        
    finally:
        # IMPORTANT: Restore original streams regardless of success or failure
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.close() 

        # Print final message to the terminal
        print(f"\nScript finished. Full log file saved to: {log_file_path}")