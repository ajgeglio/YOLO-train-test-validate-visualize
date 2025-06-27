# GobyFinder GUI Instruction Manual

## Overview
The GobyFinder GUI is a tool for running YOLO-based inference on images. It allows users to configure settings, select models, and process images through an easy-to-use interface.

---

## Prerequisites
**Python Installation**:
Ensure Python 3.8 or higher is installed on your system. You can download it from https://www.python.org/downloads/.

---

## Setting Up the environment with windows batch file

**Create the Virtual Environment on windows**:
- Double click the `create_venv.bat` which is a windows batch file script with full instructions that will install the virtual environment:

**After clicking this batch file**
- A command prompt window should pop up showing the packages being installed.
- A virtual environment called Gobyfinder will be storied in the `GobyFinderEnv` folder.
- The installation will take 10 to 15 minutes depending on the setup.

## Manual installation of the environment

**Make sure to install the GobyFinder environemnt in the same folder as the exe**
- python -m venv GobyFinderEnv
- source GobyFinderEnv\Scripts\activate
- python -m pip install ultralytics shapely

**For running gpus on windows machines that need cuda 11.8 specific cuda drivers** 
- python -m pip uninstall -y torch torchvision
- python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

---

## Using the GUI

### Text fields

**Image Directory**: Click "Browse" to select the folder containing the images you want to process.
   - Note: If you have associated labels, please follow the directory guidelines.

**Model (.pt)**: Click "Browse" to select the trained YOLO model weights file (`*.pt`) for inference.

**Output Name**: Specify the name of the output file (e.g., `results_2025`).

**Batch Settings**
- Set "Batch size" to control the number of images processed per batch.
- Set "Starting batch" to resume processing from a specific batch.
   - Note: If you plan to do a large inference run and need to pick up where you left off, you have to keep the batch size the same for the entire run.

**Confidence Threshold**: Set the minimum confidence for predictions usually ranges from 0.2 to 0.5.

**Image Size**: Specify the longest pixel dimension of the images used during training (e.g., 3008 for GoPro images and 2048 for auv images).

### Radio Button Options 

**Save image overlays**: Check to save annotated images.

**Has YOLO object labels**: Check if your images have YOLO label ground truths.
   - Note: must be saved in a folder caled "labels".

**"Has quadrat cage labels**: Check if your images have quadrate cages labeled for each image.
   - Note: must be saved in a folder called "cages" next to "images".
   
### Optional checks

3. **Check CUDA (optional)**:
   Click "Check CUDA" to verify if a CUDA-compatible GPU is available.

4. **Test YOLO (optional)**:
   Click "Test YOLO" to run a mock inference test and verify the setup.

### Execute YOLO Inference 

5. **Run Inference**:
   Click the "Run Inference" button to start processing images. The console output will display progress and logs.

---
## Directory Guidelines

**The GobyFinder GUI executable is programmed to look in its current directory for the environment.** Ensure the folder structure is as follows:

```
Folder
├── GobyFinderEnv
│   ├── ...
├── GobyFinder.exe
```

**Image and label directories should be stored together in the same folder, and the label files must have the same name as their associated images.** Use the following structure:

```
Folder
├── images
│   ├── file1.png
│   ├── file2.png
│   ├── ...
│   ├── fileN.png
├── labels
│   ├── file1.txt
│   ├── file2.txt
│   ├── ...
│   ├── fileN.txt
├── cages
│   ├── file1.txt
│   ├── file2.txt
│   ├── ...
│   ├── fileN.txt
```

This structure ensures proper functionality of the GobyFinder GUI.


---

## Troubleshooting

**Environment Not Found**:
   If the GUI shows an error about the environment, ensure you have run `create_venv.bat`.

**Manual environment troubleshooting guidance**

**Launch command prompt**:
   windows --> cmd

**Activate the Virtual Environment**:
   If the virtual environment was created, activate it by typing the path to the activation script in cmd:
   ```
   GobyFinderEnv\Scripts\activate
   ```
**Verify the Setup**:
   Check that the required dependencies are installed.
   Ensure packages like `torch`, `torchaudio`, `tkinter`, and others are listed.
   ```
   python -m pip list
   ```

**Missing Dependencies**:
   If a dependency is missing, like torch, install it manually:
   ```
   python -m pip install <package-name>
   ```
**CUDA Issues**:
   If CUDA is not detected, ensure the correct drivers and CUDA Toolkit are installed. You can also run inference on the CPU by ensuring `torch` is installed without GPU support.

**File Not Found Errors**:
   Ensure the paths to the image directory, model file, and output file are correct.

---

## Logs and Debugging
**Logs**:
- Errors are logged in `error_log.txt`.
- All other terminal output including YOLO and batch progress are saved in `log.txt`.

**Debugging**:
  If the GUI crashes or behaves unexpectedly, check the logs for details and ensure all prerequisites are met.

---

## Exiting the GUI
To exit the GUI, simply close the application window.

---

## Additional Notes
- If the cuda checks or YOLO test continues to fail, try to delete and re-install the environment.
- For large datasets, adjust the batch size to balance performance and memory usage.
- For further information on the model, refer to the Ultralytics documentation on YOLOv8.
