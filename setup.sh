#!/usr/bin/env bash
set -e

echo "=================================================="
echo "INDIAN LEGAL AI - ENVIRONMENT SETUP (LINUX)"
echo "=================================================="

# 1. Check Python
PYTHON_CMD="python3"
if ! command -v "$PYTHON_CMD" &> /dev/null; then
    echo "[ERROR] Python 3 is not installed or not in PATH."
    exit 1
fi
echo "[1/9] Python detected: $($PYTHON_CMD --version)"

# 2. Virtual Environment
if [ ! -d ".venv" ]; then
    echo "[2/9] Creating virtual environment (.venv)..."
    $PYTHON_CMD -m venv .venv
else
    echo "[2/9] Virtual environment (.venv) exists. Skipping creation."
fi

# Activate venv
source .venv/bin/activate

# 3. Dependencies
echo "[3/9] Checking dependencies..."
if command -v uv &> /dev/null; then
    uv pip install -r requirements.txt
else
    pip install -r requirements.txt
fi

# 4. PyTorch
echo "[4/9] Verifying PyTorch..."
python -c "import torch; print('PyTorch Version:', torch.__version__)"

# 5 & 6. CUDA & GPU
echo "[5/9] Checking CUDA and GPU..."
python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('Device Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

# 7. Tesseract
echo "[7/9] Checking Tesseract OCR..."
if command -v tesseract &> /dev/null; then
    echo "[✓] Tesseract is available: $(tesseract --version | head -n 1)"
else
    echo "[!] Tesseract executable not found. Install via 'sudo pacman -S tesseract' or 'sudo apt-get install tesseract-ocr'."
fi

# 8. PyMuPDF
echo "[8/9] Verifying PyMuPDF..."
python -c "import pymupdf; print('[✓] PyMuPDF is functional.')"

# 9. KB Index
echo "[9/9] Building Knowledge Base index..."
python scripts/build_index.py

echo "=================================================="
echo "[✓] SETUP COMPLETE!"
echo "Run './download.sh' to ensure model is present."
echo "Run './run.sh' to start Streamlit."
echo "=================================================="
