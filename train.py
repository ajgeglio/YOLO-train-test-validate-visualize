import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from ultralytics import YOLO, checks
import torch
from timeit import default_timer as stopwatch
import time
import argparse

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
    parser.add_argument('--note', default="training run", help='Additional notes to append on the training run')
    parser.set_defaults(resume=False)
    return parser.parse_args()

def setup_environment():
    """Ensure the bundled certificates are used."""
    os.environ['REQUESTS_CA_BUNDLE'] = os.path.join(os.path.dirname(__file__), 'certifi', 'cacert.pem')
    os.environ['SSL_CERT_FILE'] = os.path.join(os.path.dirname(__file__), 'certifi', 'cacert.pem')

def display_cuda_info():
    """Display CUDA device information."""
    device_count = torch.cuda.device_count()
    for d in range(device_count):
        print(torch.cuda.get_device_name(d))
    print("Number of devices:", device_count)

def setup_logging(training_run_folder, name_time):
    """Set up logging to a file."""
    log_file_path = os.path.join(training_run_folder, f"{name_time}_yolo_training.log")
    sys.stdout = open(log_file_path, 'w')
    return log_file_path

def main():
    setup_environment()
    args = parse_arguments()
    torch.cuda.empty_cache()
    checks()
    # Display CUDA information
    display_cuda_info()

    # Initialize variables
    print(args.note)
    start_time = stopwatch()
    torch.cuda.empty_cache()
    name_time = time.strftime("%Y%m%d%H%M", time.localtime())
    training_run_folder = os.path.join("output", "training", args.output_name)
    os.makedirs(training_run_folder, exist_ok=True)

    # Set up logging
    log_file_path = setup_logging(training_run_folder, name_time)
    print(f"Logging to: {log_file_path}")
    print(args.note)
    print(name_time)

    # Load the YOLO model
    model = YOLO(args.weights)

    # Train the model
    model.train(
        data=args.data_yml,
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
        single_cls=True,
        optimizer='auto',
        cache=False,
        exist_ok=True
    )

    # Print total training time
    print(f"Total training time: {stopwatch() - start_time:.2f} seconds")

if __name__ == '__main__':
    main()