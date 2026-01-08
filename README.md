# Name
Goby Finder with YOLO

# Description
This is a codebase for the Ultralytics YOLOv8x object detector, trained on images of Round Goby in AUV images, and to generate a proxy of size and quantity of fish in AUV data.

# Project status
In Progress

---

## scripts/ — Folder organization

The `scripts/` folder contains the main user processing interface for executing the source code in the src folder. Scripts have been organized into focused subfolders and a set of top-level helper scripts. The structure below reflects the current layout discovered in the repository and short descriptions of each script or folder's functionality.

Note: repository search results were limited and may be incomplete. View the scripts folder in the GitHub UI for the most up-to-date listing:
https://github.com/ajgeglio/YOLO-train-test-validate-visualize/tree/main/scripts

Tree (current):
```
scripts/
├─ classification/
│  ├─ rfClassify.py            # Random Forest classification helper / inference script
│  ├─ trainRF.py               # Train Random Forest classifier
│  └─ validateCLS.py           # Validation script for classification models
├─ detect/
│  ├─ batchpredict.py          # Core YOLO batched inference + CSV outputs (predictions/labels)
│  ├─ predict+overlay.py       # Single-batch inference and overlay generation (color-coded if labels)
│  ├─ runResults.py            # Pipeline launcher / wrapper that calls batchpredict, summarizes results and object sizing managing the arguments and outputing the summary reports. Requires metadata to generate reports.
│  ├─ train.py                 # Training wrapper for YOLO (resuming, args, logging)
│  └─ validate.py              # Validation wrapper (precision/recall/mAP calculations)
├─ label processing/           # (directory present for label-processing helpers)
├─ metadata processing/        # (directory present for metadata-processing helpers)
├─ transect analysis/          # (directory present for transect analysis helpers)
├─ unittest/                   # (unit tests / test scripts)
├─ visualizing/                # (visualization and plotting helpers)
├─ removeLabelDuplicates.py    # Remove duplicate lines from YOLO label .txt files
├─ cleanFilterLabels.py        # Filter and clean label reports; remove small/invalid objects, backup originals
├─ runTransect.py              # Run transect summary/report pipeline (merges results, generates summaries)
├─ labelingTransect.py         # Subsampling / create lists for manual labeling; compare subsampled biomass
```

Folder summaries and key scripts

- scripts/detect
  - batchpredict.py
    - Performs the main YOLO inference pipeline in batches.
    - Saves raw model outputs to CSV (predictions) and can optionally save label comparisons and overlays.
    - Supports SAHI tiled inference for large images.
  - predict+overlay.py
    - Minimal wrapper for running inference on a batch and creating overlay images.
    - Color-codes overlays when ground-truth labels are provided.
  - runResults.py
    - CLI launcher that parses arguments and orchestrates running batch inference and results post-processing report which includes object sizing based on auv altitude.
  - train.py
    - Training loop wrapper for YOLOv8 (resume capability, arg parsing, logging).
  - validate.py
    - Validation wrapper that computes precision/recall/mAP and saves validation outputs.

- scripts/classification
  - trainRF.py, rfClassify.py, validateCLS.py
    - Tools for extracting features, training, and validating Random Forest classifiers used for post-processing/classification tasks.

- scripts/label processing
  - removeLabelDuplicates.py
    - Scans a label folder and removes duplicate lines within each YOLO `.txt` label file (preserves order).
  - cleanFilterLabels.py
    - Loads label reports, applies pixel- and weight-based filters, and removes unwanted label lines from .txt files (creates backups when enabled).

- scripts/transect analysis
  - runTransect.py
    - High-level script to generate transect reports by merging inference results, metadata, and operation tables; produces transect summary outputs.
  - labelingTransect.py
    - Creates subsampled image lists for manual labeling and compares biomass estimates between full vs subsampled transect.

- scripts/visualizing
  - Contains utilities and scripts for visualizing predictions, PR curves, and other plots used in reporting.
  
- scripts/metadata processing
  - Prepare metadata csv for inference.

**Also supports [SAHI](https://github.com/obss/sahi) tiled inference for large images.**


## Command-Line Parameters

The following arguments can be used when running the scripts.

| Category     | Parameter         | Description                                                         | Supported In |
|--------------|-------------------|---------------------------------------------------------------------|--------------|
| **Common**   | `--weights`       | Path to model weights (.pt).                                        | All          |
|              | `--output_name`   | Name of the subfolder for results/logs.                             | All          |
|              | `--batch_size`    | Number of images per batch (use -1 for AutoBatch in Train).         | All          |
| **Training** | `--data_yml`      | Path to the dataset configuration file.                             | Train/Val    |
|              | `--epochs`        | Total number of training iterations.                                | Train        |
|              | `--patience`      | Epochs to wait for improvement before early stopping.               | Train        |
| **Inference**| `--img_directory` | Directory of images for processing.                                 | Predict      |
|              | `--sahi_tiled`    | Enables Sliced Aided Hyper Inference for large images.              | Predict      |
|              | `--confidence`    | Minimum confidence for a detection to be recorded.                  | Predict/Val  |
| **Results**  | `--metadata`      | Path to the metadata CSV for post-processing.                       | Predict      |
|              | `--has_labels`    | Compare detections against ground truth labels.                     | Predict      |

### Expected Outputs

The pipeline generates various files based on the flags provided.

| Output Type | Filename / Location | Description |
| --- | --- | --- |
| **Predictions** | `predictions.csv` | Raw model output including coordinates and confidence. |
| **Labels** | `labels.csv` | Processed ground truth labels used for comparison. |
| **Scoring** | `scores.csv` | Accuracy metrics generated when `--has_labels` is used. |
| **Results** | `inference_results_{conf}.csv` | Final output table including metadata and substrate data. |
| **Overlays** | `/overlays` folder | Visualized detection boxes drawn over original images. |
| **Logs** | `run_log_{timestamp}.log` | Detailed execution log including parameters and timestamps. |

---

### Implementation Notes

* **SAHI Tiling**: When `--sahi_tiled` is enabled, images are automatically sliced into overlapping tiles to improve small object detection on large frames.
* **Auto-Sizing**: The script includes logic to automatically adjust tile sizes and overlaps for specific image resolutions, such as `3000x4096` or `2176x4096`.
* **Feature Support**: All standard features—including label comparison, plotting, and cage calculations—are fully compatible with SAHI tiled inference.

## runResults.py


## predict+overlay.py
Minimal script for applying YOLO weights to a single batch of data and generating overlays. If labels are provided, overlays are color-coded for true/false positives.

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

## batchpredict.py
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

## GobyFinder_gui.py
This Python file provides a GUI for running inference, testing, and CUDA checks. It uses Tkinter for the interface and allows users to configure parameters for YOLO inference.

    Features:
    ----------
        - Browse for image directories and weights files.
        - Configure inference parameters such as batch size, confidence threshold, and image size.
        - Run inference, test YOLO, and check CUDA availability.
        - Real-time console output and logging.

# Source code and helper functions

## Source code and helper functions (src/)

The repository's core functionality and helper utilities live in the `src/` package. Below is an updated map of the primary source files and a short description of each so the README matches the actual codebase.

- src/utils.py  
  General utilities: file/folder helpers, image verification and dimension extraction, logging (Logger), timestamp formatting (ReturnTime), DataFrame helpers, and miscellaneous helpers used across scripts.

- src/predicting.py  
  PredictOutput: helpers to parse YOLO result objects, write predictions and labels to CSV, optionally plot overlays, and manage incremental outputs for large inference runs.

- src/results.py  
  YOLOResults (and supporting result-processing classes): merges YOLO outputs with metadata and substrate predictions, cleans/aligns columns, computes per-image statistics, calculates pixel size/area and object sizing, and estimates fish weight. (If your workflows reference an LBLResults class, confirm its exact name/location in this file.)

- src/reportFunctions.py  
  Reports: reporting and scoring utilities. Functions to generate summary prints, save/load prediction/label CSVs, compute basic counts/metrics, and orchestrate label-result merges used by higher-level scripts.

- src/fishScale.py  
  FishScale: optical geometry and size-to-weight conversion utilities (AFOV, HFOV, pixel-size/GSD, diagonal length px↔mm, corrected diagonal calculations, and weight estimation functions).

- src/image_area.py  
  ImageArea: complementary image-area and pixel-size functions used for HFOV/PS/DL calculations and weight/size conversions.

- src/labelUtils.py  
  Conversion utilities: convert COCO-like JSON or mask-based annotations to YOLO `.txt` labels and other label-format helpers used when creating training/validation label sets.

- src/dataFormatter.py  
  YOLODataFormatter: transforms raw YOLO outputs and label files into structured Pandas DataFrames (predictions and labels) ready for scoring, plotting, and CSV export.

- src/calculateIOU.py (and related iou utilities)  
  IoU calculation utilities for axis-aligned boxes (and OBB variants where present). Used by scoring/reporting code to compute IoU between labels and predictions.

- src/samUtils.py  
  SAM integration helpers (ultralytics SAM): provides SamOverlays utilities to create masks/overlays from boxes or points using a loaded SAM model.

- src/transects.py  
  Transect analysis helpers: distance-along-track calculations, filtered biomass/biomass density calculations, plotting helpers used by transect scripts like `labelingTransect.py` and `runTransect.py`.
Contains the `Reports` class for generating evaluation metrics, summaries, and plots for YOLO predictions:


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
        pip install -U ultralytics requests==2.27.1 sahi 

### 1C. Install with setup.bat (windows only)
double click setup/setup.bat

### 1D. Install with setup.py (should work for windows and linux)
python setup/setup.py

### 1D. Manual Install with PIP (If you are unable to complete step 1)
Install requirements.txt with pip (pip install -r requirements.txt)
        
        pip install -U ultralytics sahi shapely
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

@article{akyon2022sahi,
  title={Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection},
  author={Akyon, Fatih Cagatay and Altinuc, Sinan Onur and Temizel, Alptekin},
  journal={2022 IEEE International Conference on Image Processing (ICIP)},
  doi={10.1109/ICIP46576.2022.9897990},
  pages={966-970},
  year={2022}
}