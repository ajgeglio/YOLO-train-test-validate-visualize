import tkinter as tk
from tkinter import filedialog, messagebox
import subprocess
import os
import sys
import datetime
import threading

def strtime():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def get_base_path():
    """Get the base path for the virtual environment (current working directory)."""
    return os.getcwd()

def get_script_base_path():
    """Get the base path for bundled scripts (temporary folder or current directory)."""
    if hasattr(sys, '_MEIPASS'):  # PyInstaller's temporary folder
        return sys._MEIPASS
    return os.getcwd()

def get_venv_python():
    """Get the Python executable path from the virtual environment in the current working directory."""
    venv_path = os.path.join(get_base_path(), "GobyFinderEnv")
    return os.path.join(venv_path, "Scripts", "python.exe") if os.name == "nt" else os.path.join(venv_path, "bin", "python")

def load_environment():
    """Check if the virtual environment is ready."""
    venv_python = get_venv_python()
    if not os.path.exists(venv_python):
        log_error(f"Python executable not found in virtual environment: {venv_python}")
        messagebox.showinfo(
            "Environment Not Found",
            "The virtual environment is not set up. Please run 'install_venv.bat' to set it up."
        )
        return False
    else:
        log(f"Using Python executable: {venv_python}")
        log("Virtual environment is ready.")
        return True

def run_subprocess(command):
    """Run a subprocess command in a separate thread and log its output to the console in real-time."""
    def target():
        try:
            # log(f"Executing command: {' '.join(command)}")
            startupinfo = None
            if os.name == "nt":
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

            process = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, startupinfo=startupinfo
            )

            while True:
                stdout_line = process.stdout.readline()
                stderr_line = process.stderr.readline()

                if stdout_line:
                    log(stdout_line.strip())
                if stderr_line:
                    log_error(stderr_line.strip())

                if not stdout_line and not stderr_line and process.poll() is not None:
                    break

            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, command)
        except FileNotFoundError as e:
            log_error(f"File not found: {e}")
            messagebox.showerror("Error", f"File not found: {e}")
        except subprocess.CalledProcessError as e:
            log_error(f"Subprocess failed with error: {e}")
            messagebox.showerror("Error", f"Subprocess failed: {e}")
        except Exception as e:
            log_error(f"Unexpected error: {e}")
            messagebox.showerror("Error", f"Unexpected error: {e}")

    thread = threading.Thread(target=target, daemon=True)
    thread.start()

def check_cuda():
    """Check if CUDA is available using a subprocess command."""
    venv_python = get_venv_python()
    command = [venv_python, "-c", "import torch; print('CUDA Available:', torch.cuda.is_available()); \
        print('CUDA Device Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No CUDA Device')"]
    try:
        run_subprocess(command)
    except Exception as e:
        messagebox.showerror("Error", f"Failed to check CUDA availability: {e}")
        log_error(f"Failed to check CUDA availability: {e}")

def test_YOLO():
    """Run the unit test."""
    venv_python = get_venv_python()
    command = [venv_python, os.path.join(get_script_base_path(), "unittester.py")]
    try:
        run_subprocess(command)
        log("Unit test running...")
    except Exception as e:
        messagebox.showerror("Error", f"Unit test failed: {e}")
        log_error(f"Unit test failed: {e}")

def run_inference():
    """Run the inference script."""
    venv_python = get_venv_python()
    img_dir = img_dir_var.get()
    weights_path = os.path.join(get_base_path(), weights_var.get())
    output_name = output_name_var.get()
    has_labels = has_labels_var.get()
    has_cages = has_cages_var.get()
    verify_images = verify_var.get()
    plot = plot_var.get()
    start_batch = start_batch_var.get()
    batch_size = batch_size_var.get()
    confidence = confidence_var.get()
    img_size = img_size_var.get()

    if not img_dir or not weights_path or not output_name:
        messagebox.showerror("Error", "Please fill in all required fields.")
        return

    command = [
        venv_python, os.path.join(get_script_base_path(), "predict.py"),
        "--img_directory", img_dir,
        "--weights", weights_path,
        "--output_name", output_name,
        "--start_batch", start_batch,
        "--batch_size", batch_size,
        "--confidence", confidence,
        "--img_size", img_size,
        "--supress_log"
    ]
    if has_labels:
        command.append("--has_labels")
    if has_cages:
        command.append("--has_cages")
    if verify_images:
        command.append("--verify")
    if plot:
        command.append("--plot")

    try:
        run_subprocess(command)
        log("Inference started successfully...")
    except Exception as e:
        messagebox.showerror("Error", f"Inference failed: {e}")

def browse_directory(var):
    directory = filedialog.askdirectory(initialdir=os.getcwd(), title="Browse Directory")
    if directory:
        var.set(directory)

def browse_file(var):
    file_path = filedialog.askopenfilename(filetypes=[("Weights File", "*.pt")])
    if file_path:
        var.set(file_path)

def save_file(var):
    file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV File", "*.csv")])
    if file_path:
        var.set(file_path)

class ConsoleOutput:
    """Redirect sys.stdout and sys.stderr to the Tkinter console_output widget."""
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, message):
        if message.strip():
            self.text_widget.insert(tk.END, message)
            self.text_widget.see(tk.END)

    def flush(self):
        pass

class Tooltip:
    """Create a tooltip for a widget."""
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tooltip_window = None
        widget.bind("<Enter>", self.show_tooltip)
        widget.bind("<Leave>", self.hide_tooltip)

    def show_tooltip(self, event):
        if self.tooltip_window:
            return
        x, y, _, _ = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 20
        self.tooltip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(tw, text=self.text, background="yellow", relief="solid", borderwidth=1, font=("tahoma", "8", "normal"))
        label.pack(ipadx=1)

    def hide_tooltip(self, event):
        if self.tooltip_window:
            self.tooltip_window.destroy()
            self.tooltip_window = None

# Main GUI logic
root = tk.Tk()
root.title("GobyFinder Inference GUI")

# Add a console output text widget
console_output = tk.Text(root, height=20, width=80, state="normal", wrap="word", bg="black", fg="white", font=("Victor Mono", 10))
console_output.grid(row=10, column=0, columnspan=4, pady=10, padx=0)
console_output.insert(tk.END, "Console output initialized...\n")
console_output.see(tk.END)

# Redirect stdout and stderr to the console_output widget after initializing the GUI
sys.stdout = ConsoleOutput(console_output)
sys.stderr = ConsoleOutput(console_output)

def log_error(message):
    with open("error_log.txt", "a") as log_file:
        log_file.write(message + "\n")
    console_output.insert(tk.END, f"{message}\n")
    console_output.see(tk.END)

def log(message):
    with open("log.txt", "a") as log_file:
        log_file.write(message + "\n")
    console_output.insert(tk.END, f"{message}\n")
    console_output.see(tk.END)

# Initialize the environment setup before launching the GUI
if not load_environment():
    sys.exit(1)

# Variables
img_dir_var = tk.StringVar()
weights_var = tk.StringVar(value="weights/GobyFinderGoPro.pt")
output_name_var = tk.StringVar()
has_labels_var = tk.BooleanVar()
has_cages_var = tk.BooleanVar()
verify_var = tk.BooleanVar()
plot_var = tk.BooleanVar()
start_batch_var = tk.StringVar(value="0")
batch_size_var = tk.StringVar(value="2")
confidence_var = tk.StringVar(value="0.3")
img_size_var = tk.StringVar(value="3008")

# Layout
tk.Label(root, text="Image Directory:").grid(row=1, column=0, sticky="e")
tk.Entry(root, textvariable=img_dir_var, width=30).grid(row=1, column=1)
browse_img_button = tk.Button(root, text="Browse", command=lambda: browse_directory(img_dir_var))
browse_img_button.grid(row=1, column=2)
Tooltip(browse_img_button, "Select the directory containing images")

tk.Label(root, text="Model (.pt):").grid(row=2, column=0, sticky="e")
tk.Entry(root, textvariable=weights_var, width=30).grid(row=2, column=1)
browse_weights_button = tk.Button(root, text="Browse", command=lambda: browse_file(weights_var))
browse_weights_button.grid(row=2, column=2)
Tooltip(browse_weights_button, "Select a YOLO weights file (*.pt)")

tk.Label(root, text="Output name:").grid(row=3, column=0, sticky="e")
tk.Entry(root, textvariable=output_name_var, width=30).grid(row=3, column=1)
save_output_button = tk.Button(root, text="Save As", command=lambda: save_file(output_name_var))
save_output_button.grid(row=3, column=2)
Tooltip(save_output_button, "Specify the output file name")

tk.Checkbutton(root, text="Save image overlays", variable=plot_var).grid(row=5, column=2, sticky="w")
tk.Checkbutton(root, text="Has YOLO object labels", variable=has_labels_var).grid(row=6, column=2, sticky="w")
tk.Checkbutton(root, text="Has quadrat cage labels", variable=has_cages_var).grid(row=7, column=2, sticky="w")

tk.Label(root, text="Starting batch:").grid(row=5, column=0, sticky="e")
tk.Entry(root, textvariable=start_batch_var, width=10).grid(row=5, column=1, sticky="w")

tk.Label(root, text="Batch size:").grid(row=6, column=0, sticky="e")
tk.Entry(root, textvariable=batch_size_var, width=10).grid(row=6, column=1, sticky="w")

tk.Label(root, text="Confidence threshold:").grid(row=7, column=0, sticky="e")
tk.Entry(root, textvariable=confidence_var, width=10).grid(row=7, column=1, sticky="w")

tk.Label(root, text="Image size:").grid(row=8, column=0, sticky="e")
tk.Entry(root, textvariable=img_size_var, width=10).grid(row=8, column=1, sticky="w")

check_cuda_button = tk.Button(root, text="Check CUDA", command=check_cuda, bg="blue", fg="white")
check_cuda_button.grid(row=5, column=3, columnspan=1, pady=5)
Tooltip(check_cuda_button, "Check if CUDA graphics card is available")

test_yolo_button = tk.Button(root, text="Test YOLO", command=test_YOLO, bg="orange", fg="white")
test_yolo_button.grid(row=6, column=3, columnspan=1, pady=5)
Tooltip(test_yolo_button, "Run a mock YOLO inference test")

run_inference_button = tk.Button(root, text="Run Inference", command=run_inference, bg="green", fg="white", font=("Arial", 12), height=1, width=15)
run_inference_button.grid(row=8, column=2, columnspan=1, pady=10)
Tooltip(run_inference_button, "Start the inference process")

# Run the application
root.mainloop()

