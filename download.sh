#!/usr/bin/env bash
set -e

echo "=================================================="
echo "INDIAN LEGAL AI - MODEL DOWNLOADER (LINUX)"
echo "=================================================="

source .venv/bin/activate
python scripts/download_model.py
echo "[✓] Model ready."
