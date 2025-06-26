# Name
Goby Finder with Yolov8

# Description
This is a codebase for the Yolov8x object detector, trained on images of Round Goby in the Esselemen AUV images, to generate a proxy of size and quantity of Round Goby in AUV data.

# Project status

In Progress

# Notebooks and Python files

## GobyFinder_gui.py
This Python file provides a GUI for running inference, testing, and CUDA checks. It uses Tkinter for the interface and allows users to configure parameters for YOLO inference.

    Features:
    ----------
        - Browse for image directories and weights files.
        - Configure inference parameters such as batch size, confidence threshold, and image size.
        - Run inference, test YOLO, and check CUDA availability.
        - Real-time console output and logging.

## train.py
This Python file contains the training loop for a YOLOv8 model. It supports resuming training, configuring parameters via argument parsing, and logging training progress.

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

## src.py
This Python file contains utility classes and functions for data processing, label generation, and evaluation. It includes methods for handling bounding boxes, calculating IoU, generating splits, and creating annotated images.

    Features:
    ----------
        - Generate YOLO labels from JSON or mask files.
        - Calculate IoU and intersection for bounding boxes.
        - Create train-test-validation splits.
        - Generate annotated images for visualization.

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

# Installation

### 1. Install with CONDA (PREFERRED)

1a. Install with Environment.yml file (PREFERRED)

        conda env create -f environment.yml
        
#### If Error - Pip subprocess error:

1b. Resume, try again by typing...

        conda env update --file environment.yml

#### If Error - TRUSTED HOST (FOR GLSC COMPUTERS): 

Configuring pip.ini for trusted host URL

Ultralytics requires PIP. You may get an error about the trusted host. To allow PIP packages to be trusted you must Configure pip.ini  
https://stackoverflow.com/questions/25981703/pip-install-fails-with-connection-error-ssl-certificate-verify-failed-certi  
a. Check where the pip is going to look for config files  

        pip config -v list
        > For variant 'global', will try loading 'C:\ProgramData\pip\pip.ini'
        > For variant 'user', will try loading 'C:\Users\ageglio\pip\pip.ini'
        > For variant 'user', will try loading 'C:\Users\ageglio\AppData\Roaming\pip\pip.ini'     
        > For variant 'site', will try loading 'C:\Users\ageglio\AppData\Local\miniconda3\pip.ini'

b. Navigate to the folder under "For variant 'global'" create a 'pip' folder and 'pip.ini' file  
c. open with text editor and add:

        [global]
        trusted-host =  pypi.python.org
                        pypi.org
                        files.pythonhosted.org

d. conda env update --file environment.yml

### 1B. Manual Install with CONDA (If you are unable to complete step 1)

        conda create --name Yolov8  
        conda activate Yolov8  
        conda install pytorch torchvision torchaudio pytorch-cuda=11.8 pillow pip -c pytorch -c nvidia -c conda-forge  
        pip install ultralytics requests==2.27.1  

### 1C. Manual Install with PIP (If you are unable to complete step 1)
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