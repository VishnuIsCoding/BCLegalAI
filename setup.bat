@echo off
setlocal enabledelayedexpansion

echo ==================================================
echo INDIAN LEGAL AI - ENVIRONMENT SETUP (WINDOWS)
echo ==================================================

:: 1. Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH. Please install Python 3.10 or 3.11.
    pause
    exit /b 1
)
for /f "tokens=2 delims= " %%v in ('python --version') do set PY_VER=%%v
echo [1/9] Python version detected: %PY_VER%

:: 2. Create virtual environment
if not exist ".venv" (
    echo [2/9] Creating virtual environment (.venv)...
    python -m venv .venv
) else (
    echo [2/9] Virtual environment (.venv) already exists. Skipping creation.
)

:: Activate virtual environment
call .venv\Scripts\activate.bat

:: 3. Install dependencies
echo [3/9] Checking and installing dependencies...
pip install --upgrade pip
pip install -r requirements.txt

:: 4. Install / Verify PyTorch with CUDA
echo [4/9] Verifying PyTorch...
python -c "import torch; print('PyTorch Version:', torch.__version__)"

:: 5 & 6. Check CUDA and NVIDIA GPU
echo [5/9] Checking CUDA and NVIDIA GPU...
python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('Device Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

:: 7. Check Tesseract
echo [7/9] Checking Tesseract OCR...
where tesseract >nul 2>&1
if %errorlevel% neq 0 (
    echo [WARNING] Tesseract executable not found in PATH.
    echo If Tesseract is installed at C:\Program Files\Tesseract-OCR, ensure it is added to your PATH.
) else (
    echo [✓] Tesseract is available in PATH.
)

:: 8. Verify PDF extraction
echo [8/9] Verifying PyMuPDF PDF extraction...
python -c "import pymupdf; print('[✓] PyMuPDF (fitz) is functional.')"

:: 9. Verify model loading and Knowledge Base
echo [9/9] Checking Knowledge Base and Model files...
python scripts\build_index.py

echo ==================================================
echo [✓] SETUP COMPLETED SUCCESSFULLY!
echo Run 'download.bat' to download the model if not already downloaded.
echo Run 'run.bat' to start the application.
echo ==================================================
pause
