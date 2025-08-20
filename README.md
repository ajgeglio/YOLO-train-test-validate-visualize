# Name
Goby Finder with Yolov8

# Description
This is a codebase for the Yolov8x object detector, trained on images of Round Goby in AUV images, and to generate a proxy of size and quantity of fish in AUV data.

# Project status

In Progress

# Python Scripts

## GobyFinder_gui.py
This Python file provides a GUI for running inference, testing, and CUDA checks. It uses Tkinter for the interface and allows users to configure parameters for YOLO inference.

    Features:
    ----------
        - Browse for image directories and weights files.
        - Configure inference parameters such as batch size, confidence threshold, and image size.
        - Run inference, test YOLO, and check CUDA availability.
        - Real-time console output and logging.

## batchpredict.py
Batch inference script for YOLOv8. Processes images in batches, saves predictions and labels to CSV, and supports overlays and cage label output.

    Parameters:
    ----------
        --img_directory: Directory of images for inference.
        --img_list_csv: CSV file listing image paths.
        --lbl_list_csv: CSV file listing label paths.
        --weights: Path to YOLO model weights.
        --output_name: Output folder name.
        --batch_size: Number of images per batch.
        --confidence: Minimum confidence threshold.
        --has_labels: If provided, compares predictions to ground truth labels.
        --has_cages: If provided, calculates fish intersection with quadrats.
        --plot: Save overlay images.
        --verify: Verify images before processing.

    Returns:
    -------
        - CSV files with predictions, labels, and scores.
        - Overlay images if requested.

## batchpredict+results.py
Runs batch prediction and then processes results, merging YOLO outputs with metadata and other tables.

    Parameters:
    ----------
        --img_directory: Directory of images for inference.
        --weights: Path to YOLO model weights.
        --output_name: Output folder name.
        --batch_size: Number of images per batch.
        --confidence: Minimum confidence threshold.
        --metadata: Path to image metadata CSV.
        --op_table: Path to operations database table.
        --results_confidence: Confidence threshold for results output.
        --has_labels: If provided, compares predictions to ground truth labels.
        --plot: Save overlay images.

    Returns:
    -------
        - CSV file with merged YOLO results and metadata.

## predict+overlay.py
Minimal script for applying YOLO weights to a small dataset and generating overlays. If labels are provided, overlays are color-coded for true/false positives.

    Parameters:
    ----------
        --img_directory: Directory of images for inference.
        --img_list_csv: CSV file listing image paths.
        --lbl_list_csv: CSV file listing label paths.
        --weights: Path to YOLO model weights.
        --output_name: Output folder name.
        --confidence: Minimum confidence threshold.
        --plot: Save overlay images.
        --has_labels: If provided, compares predictions to ground truth labels.

    Returns:
    -------
        - Overlay images and CSVs with predictions and scores.


## train.py
This Python file contains the training loop for a YOLOv8 or other Ultralytics YOLO model. It supports resuming training, configuring parameters via argument parsing, and logging training progress.

    Parameters:
    ----------
        --weights: Initial weights for training (default: yolov8x.pt).
        --data_yaml: Path to the dataset YAML file.
        --resume: Resume training from the last checkpoint.
        --batch_size: Batch size for training.
        --epochs: Maximum number of training epochs.
        --patience: Number of epochs to wait after the best validation loss.
        --img_size: Maximum image dimension.

    Returns:
    -------
        - Logs of training progress.
        - Trained model weights saved in the specified output directory.

## predict.py
This Python file performs inference using a YOLOv8 model. It supports batch processing, label comparison, and optional plotting of predictions.

    Parameters:
    ----------
        --img_directory: Directory of images for inference.
        --weights: Path to the YOLO model weights.
        --batch_size: Number of images to process in a batch.
        --confidence: Minimum confidence threshold for detections.
        --plot: Option to save images with overlaid predictions.

    Returns:
    -------
        - CSV files containing predictions and labels (if provided).
        - Optional annotated images with bounding boxes.

## unittester.py
This Python file contains unit tests for the YOLO inference pipeline. It uses the `unittest` framework to test argument parsing, image loading, and YOLO model predictions.

    Features:
    ----------
        - Mock testing for argument parsing and file operations.
        - Validation of YOLO model predictions and outputs.

## validate.py
This Python file validates a YOLOv8 model on a dataset with existing labels. It calculates precision, recall, and mAP metrics and saves validation results.

    Parameters:
    ----------
        --data_yaml: Path to the dataset YAML file.
        --weights: Path to the YOLO model weights.
        --split: Dataset split to validate (train, test, val).
        --batch_size: Number of images per batch.
        --confidence: Minimum confidence threshold for detections.

    Returns:
    -------
        - Validation metrics (precision, recall, mAP).
        - CSV files with validation results and curves.

# Source code and helper functions

## src/utils.py

This file provides a collection of utility functions and classes for file management, image verification, logging, and data processing. Key components include:

- **ReturnTime**: Static methods for converting timestamps to formatted date/time strings.
- **Utils**: Static methods for:
    - File and folder listing, filtering, and copying/moving.
    - Image verification and dimension extraction.
    - Logging setup.
    - Generating unique image names and timestamped folders.
    - DataFrame creation for images and file operations.
    - Counting and analyzing label/object discrepancies.
    - Miscellaneous helpers for working with image and label datasets.

## src/predicting.py

Contains the `PredictOutput` class for handling YOLO prediction outputs, label processing, and cage intersection analysis:

- **PredictOutput**:
    - `YOLO_predict_w_outut`: Processes YOLO detection results, saves predictions/labels to CSV, and optionally plots overlays.
    - `YOLO_predict_w_outut_obb`: Similar to above, but for oriented bounding boxes (OBB).
    - `process_labels`: Loads and parses label files into DataFrames.
    - `intersection_df`: Calculates intersection between fish and cage bounding boxes, marking if fish are inside cages.
    - `save_cage_box_label_outut`: Saves intersection analysis results for predictions and ground truth labels.

## src/results.py

Provides the `YOLOResults` class for merging and processing YOLO inference results, metadata, substrate predictions, and survey operations tables:

- **YOLOResults**:
    - `combine_meta_pred_substrate`: Merges YOLO predictions, substrate, and metadata, aligning on filenames and survey IDs.
    - `clean_yolo_results`: Cleans and aligns columns, assigns detection IDs, and computes per-image statistics.
    - `indices`: Returns indices for camera/drone types and survey splits.
    - `area_and_pixel_size`: Calculates image area and pixel size for each detection.
    - `calc_fish_wt` / `calc_fish_wt_corr`: Estimates fish weight from bounding box size and calibration factors.
    - `yolo_results`: Main pipeline for producing cleaned, merged, and annotated YOLO results.

## src/reports.py

Contains the `Reports` class for generating evaluation metrics, summaries, and plots for YOLO predictions:

- **Reports**:
    - `generate_summary`: Prints summary statistics for predictions and labels.
    - `scores_df` / `scores_df_obb`: Computes precision, recall, and IoU for axis-aligned and oriented bounding boxes.
    - `return_fn_df`: Identifies false negatives and merges predictions with ground truths.
    - `calc_AP`, `calculate_f1`, `coco_mAP`: Calculates average precision, F1 scores, and COCO-style mAP.
    - `plot_PR`, `plot_epoch_time`, `plot_collect_distribution`: Visualization utilities for PR curves, epoch times, and data distributions.
    - `get_metrics`: Loads and summarizes metrics from saved PR curve CSVs.

These helper modules are used throughout the codebase to streamline data handling, output management, evaluation, and reporting.

# Installation

### 1. Install with CONDA (PREFERRED)

### 1a. Install with Environment.yml file (PREFERRED)

        conda env create -f environment.yml
        
#### If Error - Pip subprocess error:


### 1B. Manual Install with CONDA (If you are unable to complete step 1)

        conda create --name Yolov8  
        conda activate Yolov8  
        conda install pytorch torchvision torchaudio pytorch-cuda=11.8 pillow pip -c pytorch -c nvidia -c conda-forge  
        pip install ultralytics requests==2.27.1  

### 1C. Install with setup.bat (windows only)
double click setup/setup.bat

### 1D. Install with setup.py (should work for windows and linux)
python setup/setup.py

### 1D. Manual Install with PIP (If you are unable to complete step 1)
Install requirements.txt with pip (pip install -r requirements.txt)
        
        pip install ultralytics shapely
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118


1. Create/activate ultralytics python venv environment

        python -m venv "/path/to/virtual/environment"
        source "/path/to/virtual/environment/Scripts/activate" (WINDOWS)
        . /path/to/virtual/environment/bin/activate (LINUX)

### To update venv from requirements.txt

    $ "/path/to/virtual/environment\Scripts\python.exe" -m pip install --ignore-installed -r requirements.txt

# References

@software{yolov8_ultralytics,
  author = {Glenn Jocher and Ayush Chaurasia and Jing Qiu},
  title = {Ultralytics YOLOv8},
  version = {8.0.0},
  year = {2023},
  url = {https://github.com/ultralytics/ultralytics},
  orcid = {0000-0001-5950-6979, 0000-0002-7603-6750, 0000-0003-3783-7069},
  license = {AGPL-3.0}
}