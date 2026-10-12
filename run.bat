@echo off
echo ==================================================
echo STARTING INDIAN LEGAL AI (WINDOWS)
echo ==================================================

if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
)

echo Starting Streamlit on http://localhost:8501 ...
streamlit run app\main.py
