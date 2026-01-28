import glob
import os
import pathlib
import sys
import pandas as pd
SCRIPT_DIR = pathlib.Path(__file__).parent if '__file__' in locals() else pathlib.Path.cwd()
sys.path.append(str(SCRIPT_DIR.parent.parent / "src"))
from utils import Utils
import matplotlib.pyplot as plt

def to_base(series):
    return series.apply(lambda x: os.path.basename(x).replace('.txt', ''))

def interannotator_agreement_pairs(reference_set, comparison_set):
    """
    Find common image names between sets.
    Compute inter-annotator agreement between two sets of labels.
    Args:
        reference_set (list): List of file paths to the reference set of labels.
        comparison_set (list): List of file paths to the comparison set of labels.
    """
    # Convert lists → Series → basenames
    ds_random = to_base(pd.Series(random_labels))
    ds_hnm = to_base(pd.Series(HNM_labels))

    # find duplicates between random and HNM
    dups_between_random_hnm = set(ds_random).intersection(set(ds_hnm))
    len(dups_between_random_hnm)

    # get paths for duplicates
    random_dup_paths = [
        p for p in random_labels
        if os.path.basename(p).replace('.txt', '') in dups_between_random_hnm
    ]
    hnm_dup_paths = [
        p for p in HNM_labels
        if os.path.basename(p).replace('.txt', '') in dups_between_random_hnm
    ]
    assert len(random_dup_paths) == len(hnm_dup_paths)
    pairs = []
    for name in dups_between_random_hnm:
        r_path = next(p for p in random_labels if os.path.basename(p).startswith(name))
        l_path = next(p for p in HNM_labels if os.path.basename(p).startswith(name))
        pairs.append((name, r_path, l_path))

    return pairs

def get_all_ious(pairs):
    all_ious = []
    for name, r_path, l_path in pairs:
        r = Utils.load_yolo(r_path)
        h = Utils.load_yolo(l_path)
        ious = Utils.match_boxes(r, h)
        all_ious.extend(ious)
    return all_ious


def plot_iou_distribution(all_ious):    
    plt.figure(figsize=(8,5))
    plt.hist(all_ious, bins=20, color='steelblue', edgecolor='black')
    plt.xlabel("IoU")
    plt.ylabel("Count")
    plt.title("Inter-Annotator IoU Distribution (Reference vs Comparison)")
    plt.grid(alpha=0.3)
    plt.show()

def print_metrics(pairs):
    all_metrics = []

    for name, r_path, l_path in pairs:
        r = Utils.load_yolo(r_path)
        h = Utils.load_yolo(l_path)

        metrics = Utils.detection_metrics(h, r, iou_thresh=0.5)
        all_metrics.append(metrics)
        df = pd.DataFrame(all_metrics)
        TP = df['TP'].sum()
        FP = df['FP'].sum()
        FN = df['FN'].sum()
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0   
    print(f"Total True Positives: {TP}")
    print(f"Total False Positives: {FP}")
    print(f"Total False Negatives: {FN}")
    print(f"Overall Precision: {precision:.4f}")
    print(f"Overall Recall: {recall:.4f}")
    print(f"Overall F1 Score: {f1_score:.4f}")

if __name__ == "__main__":
    # Get the pairs of paths with duplicate images labeled in both sets
    random_labels = glob.glob(r"Z:\__Organized_Directories_InProgress\GobyFinderDatasets\AUV_datasets\innodata_2025\Random Relabel Update\untiled\labels\*.txt")
    HNM_labels = glob.glob(r"Z:\__Organized_Directories_InProgress\GobyFinderDatasets\AUV_datasets\innodata_2025\HNM Relabel Update\untiled\labels\*.txt")
    pairs = interannotator_agreement_pairs(random_labels, HNM_labels)
    print(f"Found {len(pairs)} common images labeled in both sets.")
    # Compute all IoUs between the two sets
    all_ious = get_all_ious(pairs)
    # Plot the IoU distribution
    plot_iou_distribution(all_ious)
    # Print overall detection metrics
    print_metrics(pairs)