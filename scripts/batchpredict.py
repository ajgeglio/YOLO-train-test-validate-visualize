from datetime import datetime
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from utils import Utils
from predictingFunctions import BatchOutputProcessor
from reportFunctions import Reports
from ultralytics import YOLO
import torch
from timeit import default_timer as stopwatch
import glob
import pandas as pd
import argparse
# Add SAHI imports
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.cv import read_image_as_pil

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Yolov8 inference, prediction and scoring for goby detection")
    parser.add_argument('--has_labels', action="store_true", help='Argument to do inference and compare with labels')
    parser.add_argument('--has_cages', action="store_true", help='Argument to calculate fish intersection with quadrats')
    parser.add_argument('--img_directory', default=None, help='Directory of Images')
    parser.add_argument('--img_list_csv', default=None, help='Path to csv list of image paths')
    parser.add_argument('--lbl_list_csv', default=None, help='Path to csv list of label paths')
    parser.add_argument('--weights', default=r"path\to\weights.pt", help='Trained weights path')
    parser.add_argument('--start_batch', default=0, type=int, help='Start at batch if interrupted')
    parser.add_argument('--plot', action="store_true", help='Argument to plot label + prediction overlay images')
    parser.add_argument('--supress_log', action="store_true", help='Suppress local terminal log')
    parser.add_argument('--output_name', default="inference_output", type=str, help='Name of the output csv')
    parser.add_argument('--batch_size', default=4, type=int, help='Batch size of n images in the inference loop')
    parser.add_argument('--iou', default=0.6, type=float, help='IoU threshold for Non-Maximum Suppression')
    parser.add_argument('--confidence', default=0.01, type=float, help='Minimum confidence to call a detection')
    parser.add_argument('--use_img_size', action='store_true', help="perform inference on images without defaulting to the weights default")
    parser.add_argument('--resume', action='store_true', help='use predictions.py file to continue')
    parser.add_argument('--verify', action="store_true", help='Verify image before processing')
    parser.add_argument('--sahi_tiled', action="store_true", help='Enable SAHI tiled inference')
    parser.add_argument('--tile_size', default=[1307, 1672], type=int, help='Tile size [slice_height, slice_widh] for SAHI tiled inference')
    parser.add_argument('--tile_overlap', default=[0.35, 0.275], type=float, help='Tile overlap ratio [overlap_height_ratio, overlap_width_ratio] for SAHI tiled inference ABISS1 default = [0.335, 0.275]')
    return parser.parse_args()

def setup_environment():
    """Ensure the bundled certificates are used."""
    os.environ['REQUESTS_CA_BUNDLE'] = os.path.join(os.path.dirname(__file__), 'certifi', 'cacert.pem')
    os.environ['SSL_CERT_FILE'] = os.path.join(os.path.dirname(__file__), 'certifi', 'cacert.pem')

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
        # --- Change made here ---
        # Use 'utf-8-sig' to automatically handle and remove the Byte Order Mark (BOM)
        with open(args.img_list_csv, 'r', encoding='utf-8-sig') as f:
            test_images = f.read().splitlines()
        # ------------------------
        assert len(test_images) > 0, "No image paths found in csv"
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
    elif args.img_list_csv and args.has_labels and not args.lbl_list_csv:
        try:
            with open(os.path.join(os.path.dirname(args.img_list_csv), "labels.txt"), 'r') as f:
                test_labels = f.read().splitlines()
        except:
            raise ValueError("Must provide a path to labels.txt file or a csv listing filepaths of labels")
    test_labels = sorted(test_labels)
    assert len(test_images) == len(test_labels), "Mismatch between images and labels"
    return test_labels

def save_predictions_to_csv(df, csv_path):
    """Save predictions to a CSV file, ensuring no duplicates."""
    df['detect_id'] = df.Filename + "_dt_" + df.index.astype('str')
    df = df.drop_duplicates(subset="detect_id")
    df.to_csv(csv_path, header=True)

def save_labels_to_csv(df, csv_path):
    """Save labels to a CSV file, ensuring no duplicates."""
    df = df.drop_duplicates(subset=['cls', 'x', 'y', 'w', 'h'])
    df = df.sort_values(by="Filename")
    df['ground_truth_id'] = df['Filename'] + "_" + df.index.astype('str')
    df.to_csv(csv_path, header=True)


def main():
    setup_environment()
    start_time = stopwatch()
    torch.cuda.empty_cache()

    # Get time for unique file naming
    now = datetime.now()
    name_time = now.strftime("%Y%m%d%H%M")

    # get args
    args = parse_arguments()
    # output_name = os.path.join(args.output_name + "_" + name_time)
    output_name = args.output_name
    # Setup paths and logging
    run_path = os.path.join("output", "test_runs" if args.has_labels else "inference", output_name)
    os.makedirs(run_path, exist_ok=True)
    plots_folder = os.path.join(run_path, "overlays") if args.plot else None
    if plots_folder:
        os.makedirs(plots_folder, exist_ok=True)
    Utils.initialize_logging(run_path, output_name, args.supress_log)

    # Load test images and labels
    image_list = return_img_list(args)
    label_list = return_lbl_list(args, image_list) if args.has_labels else None
    imh, imw = Utils.get_shape_pil(image_list[0])


    # Verify images if required
    if args.verify:
        image_list = Utils.verify_images(image_list)

    # Initialize YOLO model
    batch_size = args.batch_size
    start_batch = args.start_batch
    if args.resume:
        # 1. Determine which original files have been completed
        # Assumes you have selected the same batch size in your resume run as the previous run
        im_complete = len(pd.read_csv(os.path.join(run_path, "predictions.csv")).Filename.unique())
        batches_complete = im_complete // batch_size
        print(f"Resuming: at batch {batches_complete} based on predictions.py.")
        start_batch = batches_complete

    # Prepare CSV files for predictions and labels
    pred_csv_path = os.path.join(run_path, "predictions.csv")
    try:
        pd.DataFrame(columns=['Filename', 'names', 'cls', 'x', 'y', 'w', 'h', 'conf', 'imh', 'imw']).to_csv(pred_csv_path, mode='x', header=True)
    except FileExistsError:
        print("Prediction CSV already exists, appending results.")
    if args.has_labels:
        lbl_csv_path = os.path.join(run_path, f"labels.csv")
        try:
            pd.DataFrame(columns=['Filename', 'names', 'cls', 'x', 'y', 'w', 'h', 'imh_l', 'imw_l']).to_csv(lbl_csv_path, mode='x', header=True)
        except FileExistsError:
            print("Label CSV already exists, appending results.")

    # Prediction loop
    n_batches = (len(image_list) - 1) // batch_size + 1
    for k in range(start_batch, n_batches):
        print('Batch =', k)
        s, e = k * batch_size, (k + 1) * batch_size
        imgs = image_list[s:e]
        lbls = label_list[s:e] if args.has_labels else None

        if args.sahi_tiled:
            model = AutoDetectionModel.from_pretrained(
                model_type='ultralytics',
                model_path=args.weights, # any yolov8/yolov9/yolo11/yolo12/rt-detr det model is supported
                confidence_threshold=args.confidence,
                device="cuda:0", # or 'cpu' if GPU is not available
            )
            for img_path in imgs:
                image = read_image_as_pil(img_path)
                imw, imh = image.size
                if (imh, imw) == (3000, 4096):
                    tile_size=[1307, 1672]
                    tile_overlap=[0.35, 0.275]
                elif (imh, imw) == (2176, 4096):
                    tile_size=[1307, 1672]
                    tile_overlap=[0.335, 0.275]
                else:
                    tile_size=args.tile_size
                    tile_overlap=args.tile_overlap
                    
                result = get_sliced_prediction(
                    image,
                    model,
                    slice_height=tile_size[0],
                    slice_width=tile_size[1],
                    overlap_height_ratio=tile_overlap[0],
                    overlap_width_ratio=tile_overlap[1],
                    postprocess_type="NMS",
                    verbose=True,
                )
                lbl = lbls.pop(0) if lbls else None
                # Convert SAHI result to YOLO format for PredictOutput
                yolo_result = result.to_coco_annotations()
                # You may need to adapt PredictOutput.YOLO_predict_w_outut to accept this format
                BatchOutputProcessor.YOLO_predict_w_outut_sahi(
                    yolo_result, lbl, img_path, pred_csv_path,
                    lbl_csv_path if args.has_labels else None,
                    plots_folder, args.plot, args.has_labels
                )
        else:
            model = YOLO(args.weights)
            image_size = model.ckpt["train_args"]["imgsz"]
            if args.use_img_size:
                image_size = imh, imw
            results = model(
                imgs,
                stream=True,
                half=True,
                iou=args.iou,
                conf=args.confidence,
                imgsz=image_size,
                classes=[0]
            )
            for r, img_path in zip(results, imgs):
                lbl = lbls.pop(0) if lbls else None
                BatchOutputProcessor.YOLO_predict_w_outut(
                    r, lbl, img_path, pred_csv_path,
                    lbl_csv_path if args.has_labels else None,
                    plots_folder, args.plot, args.has_labels
                )
    # Finalize predictions and labels
    pred = pd.read_csv(pred_csv_path, index_col=0, low_memory=False)
    lbl = pd.read_csv(lbl_csv_path, index_col=0) if args.has_labels else None
    save_predictions_to_csv(pred, pred_csv_path)
    
    if args.has_labels:
        save_labels_to_csv(lbl, lbl_csv_path)
        df_pred, df_lbls = Reports.generate_summary(pred_csv_path, lbl_csv_path)
        df_scores = Reports.scores_df(df_lbls, df_pred, iou_tp=0.5)
        df_scores.to_csv(os.path.join(run_path, "scores.csv"))
        fndf = Reports.return_fn_df(df_lbls, df_pred, conf_thresh=args.confidence)
        fndf = fndf[fndf.fn==1]
        fndf.to_csv(os.path.join(run_path, f"false_negatives_conf_thresh_{args.confidence}.csv"), index=False)

    # Handle cages if required
    if args.has_cages:
        BatchOutputProcessor.save_cage_box_label_outut(pred, lbl, args.img_directory, run_path)

    print("Total Time: {:.2f} seconds".format(stopwatch() - start_time))
    print("Done!")

if __name__ == '__main__':
    # Example usage:
    # python scripts/predict.py --img_directory path/to/images --weights path/to/model.pt --output_name my_output --batch_size 8 --confidence 0.3 --has_labels 
    # python scripts/predict.py --img_list_csv path/to/labels.csv --lbl_list_csv path/to/labels.csv --weights path/to/model.pt --output_name my_output --batch_size 8 --confidence 0.3 --has_labels 
    main()

