import pathlib
import sys
import os
import argparse
from batchpredict import run_batch_inference, output_score_reports
SCRIPT_DIR = pathlib.Path(__file__).parent if '__file__' in locals() else pathlib.Path.cwd()
sys.path.append(str(SCRIPT_DIR.parent.parent / "src"))
from results import LBLResults, YOLOResults 

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run YOLO batch prediction and results script.")
    # Batch Inference arguments
    parser.add_argument('--directory', help='Output directory for results and inference. If not specified, will go to default location in repo output folder')
    parser.add_argument('--output_name', default="inference_output", type=str, help='Name of the output csv and folder name')
    parser.add_argument('--has_labels', action="store_true", help='Argument to do inference and compare with labels')
    parser.add_argument('--has_cages', action="store_true", help='Argument to calculate fish intersection with quadrats')
    parser.add_argument('--img_directory', default=None, help='Directory of Images')
    parser.add_argument('--img_list_file', default=None, help='Path to .txt or .csv list of image paths')
    parser.add_argument('--lbl_list_file', default=None, help='Path to .txt or .csv list of label paths, only use if it is not in the same directory as the images.txt')
    parser.add_argument('--weights', help='any yolov8/yolov9/yolo11/yolo12/rt-detr trained detect model is supported')
    parser.add_argument('--start_batch', default=0, type=int, help='Start at batch if interrupted')
    parser.add_argument('--plot', action="store_true", help='Argument to plot label + prediction overlay images')
    parser.add_argument('--suppress_log', action="store_true", help='Suppress local terminal log')
    parser.add_argument('--batch_size', default=4, type=int, help='Batch size of n images in the inference loop')
    parser.add_argument('--iou', default=0.6, type=float, help='IoU threshold for Non-Maximum Suppression')
    parser.add_argument('--confidence', default=0.01, type=float, help='Minimum confidence to call a detection')
    parser.add_argument('--use_img_size', action='store_true', help="perform inference on images without defaulting to the weights default")
    parser.add_argument('--resume', action='store_true', help='use predictions.py file to continue')
    parser.add_argument('--verify', action="store_true", help='Verify image before processing')
    parser.add_argument('--sahi_tiled', action="store_true", help='Enable SAHI tiled inference')
    parser.add_argument('--tile_size', default=[1307, 1672], type=int, help='Tile size [slice_height, slice_widh] for SAHI tiled inference')
    parser.add_argument('--tile_overlap', default=[0.35, 0.275], type=float, help='Tile overlap ratio [overlap_height_ratio, overlap_width_ratio] for SAHI tiled inference ABISS1 default = [0.335, 0.275]')
    # Additional arguments for results output
    parser.add_argument("--use_predictions", action="store_true", help="Whether to use already run prediction results.")
    parser.add_argument("--results_confidence", type=float, default=0.2, help="Results confidence threshold")
    parser.add_argument("--op_table", default=r"Z:\__AdvancedTechnologyBackup\07_Database\OP_TABLE.xlsx", help="Path to OP_TABLE.xlsx")
    parser.add_argument("--metadata", required=True, help="Path to meta.csv")
    parser.add_argument("--substrate", help="Optional substrate inference results path") 
    return parser.parse_args()

def process_results(args, run_path):
    """Logic moved from batchpredict+results.py"""
    yolo_infer_path = os.path.join(run_path, "predictions.csv")
    output = YOLOResults(args.metadata, yolo_infer_path, args.substrate, args.op_table, args.results_confidence)
    yolores = output.yolo_results(find_closest=False)
    
    out_path = os.path.join(run_path, f"inference_results_{args.results_confidence:04.2f}.csv")
    yolores.to_csv(out_path, index=False)

    if args.has_labels:
        lbl_pth = os.path.join(run_path, "labels.csv")
        output_lbl = LBLResults(args.metadata, lbl_pth, args.substrate, args.op_table)
        lbl_res = output_lbl.lbl_results(find_closest=False)
        lbl_res.to_csv(os.path.join(run_path, "label_box_results.csv"), index=False)


def run_results(args):
    # 1. Setup Paths
    if args.directory:
         run_path = args.directory
    else:
        run_path = os.path.join("output", "test_runs" if args.has_labels else "inference", args.output_name)
    os.makedirs(run_path, exist_ok=True)

    # 2. Run Inference (Modular Call)
    if not args.use_predictions:
        print(f"Starting Inference: {args.output_name}")
        assert args.weights, "You must provide YOLO weights to run inference"
        print(f"using weights path", args.weights)
        run_batch_inference(args) # Calls the engine directly

    # 3. Process Results (Logic moved from batchpredict+results.py)
    yolo_infer_path = os.path.join(run_path, "predictions.csv")
    if os.path.exists(yolo_infer_path):
        output = YOLOResults(args.metadata, yolo_infer_path, args.substrate, args.op_table, args.results_confidence)
        yolores = output.yolo_results(find_closest=False) #
        
        out_path = os.path.join(run_path, f"inference_results_{args.results_confidence:04.2f}.csv")
        yolores.to_csv(out_path, index=False)
    
    if args.has_labels:
        lbl_pth = os.path.join(run_path, "labels.csv")
        lbl_output = LBLResults(args.metadata, lbl_pth, args.substrate, args.op_table)
        lblres = lbl_output.lbl_results(find_closest=False) #
        lblres.to_csv(os.path.join(run_path, "label_box_results.csv"), index=False)

        ## If we have filtered predicitons, re-run the reports
        if os.path.exists(os.path.join(run_path, "predictions_filtered.csv")):
            pred_csv_path = os.path.join(run_path, "predictions_filtered.csv")
            lbl_csv_path = os.path.join(run_path, "labels.csv")
            output_score_reports(pred_csv_path, lbl_csv_path, run_path, confidence_thresh=args.confidence)

if __name__ == "__main__":
    args = parse_arguments()
    run_results(args)