# ultralytics
# shapely

# Note: Ensure the correct PyTorch version with CUDA support is installed.
# Use the following command to install PyTorch with CUDA 11.8 support:
# pip uninstall torch torchvision
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

pyinstaller --onefile --noconsole --icon=icon.png \
    --add-data "GobyFinder/predict.py;." \
    --add-data "GobyFinder/src.py;." \
    --add-data "GobyFinder/unittester.py;." \
    --add-data "C:\Users\ageglio\AppData\Local\Programs\Python\Python313\tcl\tcl8.6;lib\tcl8.6" \
    --add-data "C:\Users\ageglio\AppData\Local\Programs\Python\Python313\tcl\tk8.6;lib\tk8.6" \
    --add-data "C:\Users\ageglio\ageglio-1\gobyfinder_yolov8\scripts\GobyFinderEnv\Lib\site-packages\certifi\cacert.pem;certifi" \
    GobyFinder/GobyFinder_gui.py

