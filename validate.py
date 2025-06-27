from ultralytics import YOLO
import argparse
from timeit import default_timer as stopwatch
import torch
import os
import pandas as pd
import numpy as np 

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description = "Yolov8 validation statistics for goby detection, existing labels needed")
    parser.add_argument('--data_yaml',dest='data_yaml',default=None, help = "yml file pointing to directory of images to perform inference on (must have labels)")
    parser.add_argument('--weights', dest="weights", default=r'src\models\GobyFinderAUV.pt', help = 'weights path')
    parser.add_argument('--split', dest="split", default='test', help = 'split to do val on yaml file: train, test, val')
    parser.add_argument('--output_name', dest='output_name', default = "validation_output", type = str, help = 'name of the ouput csv, also the folder created in the validation folder')
    parser.add_argument('--batch_size', dest='batch_size', default=2, type=int, help = 'Sets the number of images per batch. Use -1 for AutoBatch, which automatically adjusts based on GPU memory availability')
    parser.add_argument('--confidence', dest='confidence', default=0.01, type=float, help = 'Sets the minimum confidence threshold for detections. Detections with confidence below this threshold are discarded.')
    parser.add_argument('--iou', dest='iou', default=0.6, type=float, help = 'Sets the Intersection Over Union (IoU) threshold for Non-Maximum Suppression (NMS). Helps in reducing duplicate detections.')
    parser.add_argument('--save_hybrid', dest='save_hybrid', action= "store_true", help = 'If True, saves a hybrid version of labels that combines original annotations with additional model predictions.')
    parser.add_argument('--img_size', dest='img_size', default=2048, type=int, help = 'max image dimension')
    return parser.parse_args()


if __name__ == '__main__':
    start_time = stopwatch()
    torch.cuda.empty_cache()

    # get args
    args = parse_arguments()

    model = YOLO(args.weights)
    folder = os.path.join("output", "validation","detect")
    project_folder = os.path.join(folder, args.output_name)
    output_csv = os.path.join(project_folder, args.output_name+".csv")
    output_curves = os.path.join(project_folder, args.output_name+"_curves.csv")
    # Validate the model
    metrics = model.val(data = args.data_yaml,
                        split = args.split,
                        device = 0,
                        half = True,
                        batch = args.batch_size,
                        project = project_folder,
                        plots = True,
                        conf = args.confidence,
                        imgsz = args.img_size,
                        iou = args.iou,
                        save_hybrid = args.save_hybrid,
                        save_json = True)
    mAP = metrics.box.map
    map50 = metrics.box.ap50  # map50
    p = metrics.box.p
    r = metrics.box.r
    f1 = metrics.box.f1
    all_ap = metrics.box.all_ap
    ar = np.c_[p, r, f1, map50, mAP]
    pd.DataFrame(ar, columns=["precision", "recall", "f1", "mAP50", "mAP"]).to_csv(output_csv)
    pd.DataFrame(metrics.curves_results).to_csv(output_curves)

'''
        Attributes:
        p (list): Precision for each class. Shape: (nc,).
        r (list): Recall for each class. Shape: (nc,).
        f1 (list): F1 score for each class. Shape: (nc,).
        all_ap (list): AP scores for all classes and all IoU thresholds. Shape: (nc, 10).
        ap_class_index (list): Index of class for each AP score. Shape: (nc,).
        nc (int): Number of classes.


    Methods:
    process(tp, conf, pred_cls, target_cls): Updates the metric results with the latest batch of predictions.
    keys: Returns a list of keys for accessing the computed detection metrics.
    mean_results: Returns a list of mean values for the computed detection metrics.
    class_result(i): Returns a list of values for the computed detection metrics for a specific class.
    maps: Returns a dictionary of mean average precision (mAP) values for different IoU thresholds.
    fitness: Computes the fitness score based on the computed detection metrics.
    ap_class_index: Returns a list of class indices sorted by their average precision (AP) values.
    results_dict: Returns a dictionary that maps detection metric keys to their computed values.
    curves: TODO
    curves_results: TODO
'''