import os
import sys
import subprocess


def get_venv_path():
    """Get the path to the virtual environment."""
    return os.path.join(os.path.dirname(__file__), "GobyFinderEnv")


def get_venv_python():
    """Get the Python executable path in the virtual environment."""
    venv_path = get_venv_path()
    return os.path.join(venv_path, "Scripts", "python.exe") if sys.platform == "win32" else os.path.join(venv_path, "bin", "python")


def create_virtual_environment(venv_path):
    """Create a virtual environment if it doesn't exist."""
    if not os.path.exists(venv_path):
        try:
            subprocess.run(["python", "-m", "venv", venv_path], check=True)
            print("Virtual environment created successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Error creating virtual environment: {e}")
            print(e.stdout)
            print(e.stderr)
            raise


def run_subprocess(command, description):
    """Run a subprocess command and handle errors."""
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print(result.stdout)
        print(result.stderr)
        print(f"{description} completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error during {description}: {e}")
        print(e.stdout)
        print(e.stderr)
        raise


def install_dependencies():
    """Install dependencies in the virtual environment."""
    venv_path = get_venv_path()
    create_virtual_environment(venv_path)

    venv_python = get_venv_python()
    if not os.path.exists(venv_python):
        print(f"Python executable not found in virtual environment: {venv_python}")
        return

    # Install ultralytics and shapely
    run_subprocess(
        [venv_python, "-m", "pip", "install", "ultralytics", "shapely"],
        "Installing ultralytics and shapely"
    )

    # Uninstall torch and torchvision
    run_subprocess(
        [venv_python, "-m", "pip", "uninstall", "-y", "torch", "torchvision"],
        "Uninstalling torch and torchvision"
    )

    # Install CUDA-specific torch dependencies
    run_subprocess(
        [venv_python, "-m", "pip", "install", "torch", "torchvision", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cu118"],
        "Installing CUDA-specific torch dependencies"
    )


def check_cuda():
    """Check if CUDA is available."""
    venv_python = get_venv_python()
    command = [venv_python, "-c", "import torch; print('CUDA Available:', torch.cuda.is_available())"]
    run_subprocess(command, "Checking CUDA availability")


if __name__ == "__main__":
    try:
        install_dependencies()
        check_cuda()
    except Exception as e:
        print(f"An error occurred: {e}")
