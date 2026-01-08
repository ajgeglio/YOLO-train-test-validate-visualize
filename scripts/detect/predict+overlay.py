from ultralytics import YOLO
import sys
import glob
import os
import pathlib
SCRIPT_DIR = pathlib.Path(__file__).parent if '__file__' in locals() else pathlib.Path.cwd()
sys.path.append(str(SCRIPT_DIR.parent.parent / "src"))
from dataFormatter import YOLODataFormatter
from overlayFunctions import Overlays
from reportFunctions import Reports
import pandas as pd
from datetime import datetime
import argparse
"""
This is a minimal script for applying YOLO weights to a small dataset (single batch) of images and generating the overlays.
If labels are provided, it will generate color coded overlays for True positive and False positive predictions.
The current functionality does not save out any csvs.
"""
def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Yolov8 inference, prediction and scoring for goby detection")
    parser.add_argument('--has_labels', action="store_true", help='Argument to do inference and compare with labels')
    parser.add_argument('--img_directory', dest='img_directory', default=None, help='Directory of Images')
    parser.add_argument('--img_list_csv', dest="img_list_csv", default=None, help='Path to csv list of image paths')
    parser.add_argument('--lbl_list_csv', dest="lbl_list_csv", default=None, help='Path to csv list of label paths')
    parser.add_argument('--img_path', dest='img_path', default=None, help='Path to a single image')
    parser.add_argument('--lbl_path', dest='lbl_path', default=None, help='Path to a single label txt file')
    parser.add_argument('--weights', dest="weights", default=r"path\to\model_weights.pt", help='Weights path')
    parser.add_argument('--plot', action="store_true", help='Argument to plot label + prediction overlay images')
    parser.add_argument('--output_name', dest='output_name', default="overlay_output", type=str, help='Name of the output csv')
    parser.add_argument('--img_size', dest='img_size', default=4096, type=int, help='Max image dimension')
    parser.add_argument('--iou', dest='iou', default=0.6, type=float, help='IoU threshold for Non-Maximum Suppression')
    parser.add_argument('--sample', default=None, type=int, help='Number of images to randomly sample if doing a subset')
    parser.add_argument('--confidence', dest='confidence', default=0.01, type=float, help='Minimum confidence to call a detection')
    parser.add_argument('--verify', action="store_true", help='Verify image before processing')
    return parser.parse_args()

def return_img_list(args):
    """Retrieve test images based on provided arguments."""
    if args.img_directory:
        img_dir = args.img_directory
        print("Testing img_directory:", img_dir)
        test_images = glob.glob(f'{img_dir}/*.[jJ][pP][gG]') + \
                      glob.glob(f'{img_dir}/*.[tT][iI][fF]') + \
                      glob.glob(f'{img_dir}/*.[pP][nN][gG]')
        assert len(test_images) > 0, "No images found in directory"
    elif args.img_list_csv:
        print("Using img_list_csv")
        with open(args.img_list_csv, 'r') as f:
            test_images = f.read().splitlines()
        assert len(test_images) > 0, "No image paths found in csv"
    elif args.img_path:
        test_images = [args.img_path]
    else:
        raise ValueError("Must provide an image directory or a path to a csv listing filepaths of images")
    return sorted(test_images)

def return_lbl_list(args, test_images):
    """Retrieve test labels if labels are provided."""
    if args.img_directory:
        dirlbl = os.path.join(os.path.dirname(args.img_directory), "labels")
        print("Label directory:", dirlbl)
        test_labels = glob.glob(os.path.join(dirlbl, "*.txt"))
    elif args.lbl_list_csv:
        with open(args.lbl_list_csv, 'r') as f:
            test_labels = f.read().splitlines()
    elif args.lbl_path:
        test_labels = [args.lbl_path]
    else:
        raise ValueError("Must provide a label directory or a list of filepaths")
    test_labels = sorted(test_labels)
    assert len(test_images) == len(test_labels), "Mismatch between images and labels"
    return test_labels

def main():
    args = parse_arguments()

    # Get time for unique file naming
    now = datetime.now()
    name_time = now.strftime("%Y%m%d%H%M")

    # Setup paths
    output_name = os.path.join(args.output_name + "_" + name_time)
    # Setup paths and logging
    run_path = os.path.join("output", "test_runs" if args.has_labels else "inference", output_name)
    os.makedirs(run_path, exist_ok=True)
    plots_folder = os.path.join(run_path, "overlays") if args.plot else None
    if plots_folder:
        os.makedirs(plots_folder, exist_ok=True)

    # Load test images and labels
    image_list = return_img_list(args)
    label_list = return_lbl_list(args, image_list) if args.has_labels else None

    random_state = 123
    sample_n_imgs = args.sample

    if not image_list:
        raise ValueError("No images found in the specified directories.")
    if args.has_labels and not label_list:
        raise ValueError("No labels found in the specified directories.")
    
    if sample_n_imgs is None:
        sample_n_imgs = len(image_list)
    if len(image_list) < sample_n_imgs:
        raise ValueError(f"Not enough images to sample {sample_n_imgs}. Found {len(image_list)} images.")
    if args.has_labels and len(label_list) < sample_n_imgs:
        raise ValueError(f"Not enough labels to sample {sample_n_imgs}. Found {len(label_list)} labels.")

    imgs = pd.Series(image_list).sample(sample_n_imgs, random_state=random_state).tolist()
    lbls = pd.Series(label_list).sample(sample_n_imgs, random_state=random_state).tolist() if args.has_labels else None

    conf_thresh = args.confidence
    model = YOLO(args.weights)
    results = model(imgs, stream=True, half=True, iou=args.iou, conf=conf_thresh, imgsz=args.img_size)
    plot = args.plot
    labels = args.has_labels

    if labels:
        df_lbl, df_prd = YOLODataFormatter.return_lbl_pred_df(results, lbls, imgs)
        df_scores = Reports.scores_df(df_lbl, df_prd, iou_tp=0.5)
        df_scores.to_csv(os.path.join(run_path, f"scores.csv"), index=False)
        df_lbl.to_csv(os.path.join(run_path, f"labels.csv"), index=False)
        df_prd.to_csv(os.path.join(run_path, f"predictions.csv"), index=False)
        print("Labels and predictions saved to csvs")
    else:
        df_prd = YOLODataFormatter.return_pred_df(results, imgs)
        df_prd.to_csv(os.path.join(run_path, f"predictions.csv"), index=False)
        print("Predictions saved to csv")
    if plot:
        if labels:
            for img_pth in imgs:
                Overlays.save_annot_imgs2(img_pth, df_scores, df_lbl, plots_folder, conf_thresh)
        else:
            for img_pth in imgs:
                Overlays.save_annot_imgs_pred_only(img_pth, df_prd, plots_folder, conf_thresh)

if __name__ == "__main__":
    main()