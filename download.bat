@echo off
echo ==================================================
echo INDIAN LEGAL AI - MODEL DOWNLOADER (WINDOWS)
echo ==================================================

if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
)

python scripts\download_model.py
if %errorlevel% neq 0 (
    echo [ERROR] Model download failed.
    pause
    exit /b 1
)

echo [✓] Model ready.
pause
