"""
Model Download Script for Indian Legal AI
Downloads GSMS-B/Indian-Legal-Qwen2.5-1.5B once and stores it locally in models/.
Checks for existing model files and resumes safely if interrupted.
"""

import os
import sys
from pathlib import Path
from huggingface_hub import snapshot_download

MODEL_REPO = "GSMS-B/Indian-Legal-Qwen2.5-1.5B"
LOCAL_DIR = Path("models/indian-legal-qwen-1.5b").resolve()

REQUIRED_FILES = [
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
]


def is_model_complete(target_dir: Path) -> bool:
    if not target_dir.exists():
        return False
    for filename in REQUIRED_FILES:
        file_path = target_dir / filename
        if not file_path.exists():
            return False
        # Verify model.safetensors is not empty or truncated
        if filename == "model.safetensors" and file_path.stat().st_size < 2_500_000_000:
            print(f"[!] Warning: {filename} size is {file_path.stat().st_size} bytes, which appears incomplete.")
            return False
    return True


def download_model(repo_id: str = MODEL_REPO, local_dir: Path = LOCAL_DIR):
    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"==================================================")
    print(f"INDIAN LEGAL AI - MODEL DOWNLOADER")
    print(f"Target Repo: {repo_id}")
    print(f"Local Path:  {local_dir}")
    print(f"==================================================")

    if is_model_complete(local_dir):
        print(f"[✓] Model already exists and is complete at: {local_dir}")
        print("No download needed.")
        return str(local_dir)

    print(f"[*] Starting download from Hugging Face ({repo_id})...")
    print("This will download ~2.87 GB of model weights. Please wait...")

    try:
        downloaded_path = snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
            resume_download=True,
            local_dir_use_symlinks=False,
        )
        print(f"[✓] Successfully downloaded {repo_id} to {downloaded_path}")

        # Verification
        if is_model_complete(local_dir):
            print("[✓] Model files integrity verified successfully.")
            return str(local_dir)
        else:
            print("[!] Warning: Download finished but some expected files are missing.")
            return str(local_dir)

    except Exception as e:
        print(f"[✗] Error downloading model: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    download_model()
