#!/usr/bin/env bash
set -e

echo "=================================================="
echo "STARTING INDIAN LEGAL AI (LINUX)"
echo "=================================================="

source .venv/bin/activate
echo "Starting Streamlit on http://localhost:8501 ..."
streamlit run app/main.py
