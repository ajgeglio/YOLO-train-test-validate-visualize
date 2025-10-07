@echo off
REM Use %USERPROFILE% to dynamically reference the user's home directory
set BASE_PATH=%USERPROFILE%

REM Define the Python executable path
set PYTHON_EXEC=%BASE_PATH%\AppData\Local\Programs\Python\Python313\python.exe

REM Check if Python is installed
"%PYTHON_EXEC%" --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python is not installed at %PYTHON_EXEC%. Please install Python and try again.
    pause
    exit /b
)

REM Create the virtual environment
"%PYTHON_EXEC%" -m venv GobyFinderEnv
if %errorlevel% neq 0 (
    echo Failed to create the virtual environment.
    pause
    exit /b
)

REM Activate the virtual environment
call GobyFinderEnv\Scripts\activate

REM Upgrade pip
python -m pip install --upgrade pip

REM Install required dependencies
python -m pip install -U ultralytics shapely sahi
python -m pip uninstall -y torch torchvision
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo Virtual environment setup complete.
pause