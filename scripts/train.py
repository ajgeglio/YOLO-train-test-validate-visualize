import sys
import os
import argparse
import time
from timeit import default_timer as stopwatch
from ultralytics import YOLO, checks
import torch
from pathlib import Path # <-- 1. ADD THIS IMPORT

# --- Logger class to tee output to both file and terminal ---
class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Yolov8 training for goby detection")
    parser.add_argument('--weights', default='yolov8x.pt', help='Initial weights for training')
    parser.add_argument('--data_yml', default=None, help='YAML file for data')
    parser.add_argument('--resume', action='store_true', help='Resume training from the last checkpoint')
    parser.add_argument('--output_name', default="training_run", type=str, help='Name of the training run')
    parser.add_argument('--batch_size', default=-1, type=int, help='Batch size for training, defaults to -1 for autobatch')
    parser.add_argument('--epochs', default=500, type=int, help='Maximum number of training epochs')
    parser.add_argument('--patience', default=16, type=int, help='Number of epochs to wait after the best validation loss')
    parser.add_argument('--img_size', default=2048, type=int, help='Maximum image dimension')
    parser.add_argument('--lr0', default=0.01, type=float, help="The starting learning rate, default same as Ultralytics")
    parser.add_argument('--lrf', default=0.01, type=float, help="Final learning rate as a fraction of the initial rate. The final LR will be lr0xlrf=0.01x0.01=0.0001.")
    parser.add_argument('--warmup_epochs', default=3, type=int, help="Number of epochs for the linear learning rate warmup phase.")
    parser.add_argument('--warmup_momentum', default=0.8, type=float, help="Initial momentum during the warmup phase.")
    parser.add_argument('--optimizer', default='auto', help="Automatically selects the optimizer (often defaults to SGD for detection/segmentation).")
    parser.add_argument('--note', default="training run", help='Additional notes to append on the training run')
    # Re-adding single_cls as a boolean flag. It should be False (default) for 6 classes.
    parser.add_argument('--single_cls', action='store_true', help='Treats all classes as a single class (only use for binary classification).') 
    parser.set_defaults(resume=False)
    return parser.parse_args()

def setup_environment():
    """Ensure the bundled certificates are used."""
    os.environ['REQUESTS_CA_BUNDLE'] = os.path.join(os.path.dirname(__file__), 'certifi', 'cacert.pem')
    os.environ['SSL_CERT_FILE'] = os.path.join(os.path.dirname(__file__), 'certifi', 'cacert.pem')

def display_cuda_info():
    """Display CUDA device information."""
    device_count = torch.cuda.device_count()
    if device_count > 0:
        for d in range(device_count):
            print(f"Device {d}: {torch.cuda.get_device_name(d)}")
    else:
        print("No CUDA devices found. Using CPU.")
    print("Number of devices:", device_count)

def main():
    start_time = stopwatch()
    
    # --- 1. Setup paths and parse arguments early ---
    args = parse_arguments()
    name_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    training_run_folder = os.path.join("output", "training", args.output_name)
    os.makedirs(training_run_folder, exist_ok=True)
    log_file_path = os.path.join(training_run_folder, f"{name_time}_yolo_training.log")

    # --- 2. POSIX Path Fix for Ultralytics ---
    # This forces the path into the format Ultralytics accepts for local files, 
    # preventing the download attempt.
    if args.data_yml:
        data_path_posix = str(Path(args.data_yml).resolve().as_posix())
    else:
        data_path_posix = None
    
    # --- 3. Set up logging to capture all subsequent output ---
    sys.stdout = Logger(log_file_path)

    print("--- Training Run Started ---")
    print(f"Log file created at: {log_file_path}")
    print(f"Start time: {time.ctime()}")

    # --- 4. Log all input arguments ---
    print("\n--- Input Arguments ---")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print(f"  data_yml (POSIX resolved): {data_path_posix}") # Log the corrected path
    print("---------------------\n")

    # --- 5. Begin core script logic ---
    setup_environment()
    torch.cuda.empty_cache()
    checks()
    display_cuda_info()
    print(f"Note: {args.note}\n")

    # Load the YOLO model
    model = YOLO(args.weights)

    # Train the model
    print("Starting YOLOv8 training...")
    model.train(
        data=data_path_posix, # <-- USE THE POSIX PATH
        # compile="default",
        resume=args.resume,
        pretrained=True,
        epochs=args.epochs,
        imgsz=args.img_size,
        device=0,
        project=training_run_folder,
        half=True,
        amp=True,
        seed=123,
        batch=args.batch_size,
        patience=args.patience,
        val=True,
        single_cls=args.single_cls,
        cache=False,
        exist_ok=True,
        optimizer=args.optimizer,
        lr0 = args.lr0,
        lrf = args.lrf,
        warmup_epochs = args.warmup_epochs,
        warmup_momentum = args.warmup_momentum,
    )

    # Print total training time
    print(f"\nTotal execution time: {stopwatch() - start_time:.2f} seconds")
    print("--- Training Run Complete ---")

if __name__ == '__main__':
    main()