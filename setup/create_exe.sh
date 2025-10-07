#!/bin/bash

# Cross-platform path separator
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" || "$OSTYPE" == "cygwin" ]]; then
    SEP=";"
else
    SEP=":"
fi

# Optional: Clean previous builds
echo "Cleaning previous builds..."
rm -rf build dist *.spec

# Build command
echo "Building executable with PyInstaller..."
pyinstaller --onefile --noconsole --icon=assets/icon.png \
    --add-data="scripts/batchpredict.py${SEP}scripts" \
    --add-data="scripts/unittester.py${SEP}src" \
    --add-data="src/utils.py${SEP}src" \
    --add-data="src/predicting.py${SEP}src" \
    --add-data="src/reports.py${SEP}src" \
    --add-data="src/iou.py${SEP}src" \
    --add-data="certifi/cacert.pem:certifi" \
    scripts/GobyFinderGui.py

# Check result
if [[ $? -eq 0 ]]; then
    echo "✅ Build successful! Executable is in the 'dist' folder."
else
    echo "❌ Build failed. Check the output above for errors."
    exit 1
fi