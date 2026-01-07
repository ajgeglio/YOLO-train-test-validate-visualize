import subprocess
import sys
import os
import argparse
from batchpredict import run_batch_inference
from results import LBLResults, YOLOResults 

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

def main():
    parser = argparse.ArgumentParser(description="Run YOLO batch prediction and results script.")
    parser.add_argument("--img_directory", required=False, help="Path to image directory")
    parser.add_argument("--img_list_csv", required=False, help="Path to image list csv")
    parser.add_argument("--weights", required=True, default=r"path\to\model_weights.pt", help="Path to YOLO model weights")
    parser.add_argument("--op_table", default=r"Z:\__AdvancedTechnologyBackup\07_Database\OP_TABLE.xlsx", help="Path to OP_TABLE.xlsx")
    parser.add_argument("--metadata", required=True, help="Path to meta.csv")
    parser.add_argument("--substrate", help="Optional substrate inference results path") 
    parser.add_argument("--output_name", required=True, help="Output name for results")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--confidence", type=float, default=0.01, help="Confidence threshold")
    parser.add_argument("--results_confidence", type=float, default=0.2, help="Results confidence threshold")
    parser.add_argument("--start_batch", type=int, default=0, help="Start batch index")
    parser.add_argument("--has_labels", action="store_true", help="Whether labels are present")
    parser.add_argument("--run_predict", action="store_true", help="Whether to run prediction")
    parser.add_argument("--plot", action="store_true", help="Whether to plot overlays")
    parser.add_argument('--resume', action='store_true', help='use predictions.py file to continue')
    parser.add_argument('--use_img_size', action='store_true', help="perform inference on image size without defaulting to the weights default")
    parser.add_argument('--sahi_tiled', action="store_true", help='Enable SAHI tiled inference')
    args = parser.parse_args()

    # 1. Setup Paths
    run_path = os.path.join("output", "test_runs" if args.has_labels else "inference", args.output_name)
    os.makedirs(run_path, exist_ok=True)

    # 2. Run Inference (Modular Call)
    if args.run_predict:
        print(f"Starting Inference: {args.output_name}")
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

if __name__ == "__main__":
    main()