# -*- coding: utf-8 -*-
"""
mistral_api_finetune.py -- Mistral Hosted Fine-Tuning API integration
======================================================================
Uploads the local `train.jsonl` and `val.jsonl` data to Mistral's servers 
and triggers a fine-tuning job using the official `mistralai` package.
"""

import os
import time
import json
from pathlib import Path
from mistralai.client import Mistral
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env", override=True)

# ── Paths ─────────────────────────────────────────────────────────────────────
SPLITS_DIR = BASE_DIR / "data" / "splits"
TRAIN_FILE = SPLITS_DIR / "train.jsonl"
VAL_FILE   = SPLITS_DIR / "val.jsonl"

def upload_file(client: Mistral, filepath: Path) -> str:
    print(f"[*] Uploading {filepath.name} to Mistral API...")
    with open(filepath, "rb") as f:
        uploaded_file = client.files.upload(
            file={
                "file_name": filepath.name,
                "content": f,
            },
            purpose="fine-tune"
        )
    print(f"[OK] Uploaded {filepath.name}. File ID: {uploaded_file.id}")
    return uploaded_file.id

def main():
    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError("MISTRAL_API_KEY environment variable not found in .env")
        
    client = Mistral(api_key=api_key)

    print("\n" + "="*55)
    print("  FUMII MISTRAL API FINE-TUNING")
    print("="*55)

    if not TRAIN_FILE.exists():
        print("[ERROR] train.jsonl not found. Please run prepare_data.py first.")
        return

    # 1. Upload Training Data
    train_file_id = upload_file(client, TRAIN_FILE)
    
    # 2. Upload Validation Data (if exists)
    val_file_id = None
    if VAL_FILE.exists() and VAL_FILE.stat().st_size > 0:
        val_file_id = upload_file(client, VAL_FILE)
    elif VAL_FILE.exists():
        print("[WARN] val.jsonl is empty. Skipping validation file upload.")
        
    # Wait a few seconds for files to be processed by Mistral's backend before creating job
    print("[*] Waiting 5 seconds for file ingestion...")
    time.sleep(5)

    # 3. Create Fine-Tuning Job
    print("[*] Triggering fine-tuning job on 'ministral-8b-latest'...")
    
    job_kwargs = {
        "model": "ministral-8b-latest",
        "training_files": [{"file_id": train_file_id, "weight": 1}],
        "hyperparameters": {
            "training_steps": 100,
            "learning_rate": 0.0001
        },
        "auto_start": True
    }
    
    if val_file_id:
        job_kwargs["validation_files"] = [val_file_id]

    try:
        created_job = client.fine_tuning.jobs.create(**job_kwargs)
        
        print("\n[SUCCESS] Fine-tuning job created successfully!")
        print(f"  Job ID       : {created_job.id}")
        print(f"  Status       : {created_job.status}")
        print(f"  Base Model   : {created_job.model}")
        print(f"  Hyperparams  : {created_job.hyperparameters}")
        
        # Save Job ID to outputs folder for tracking
        outputs_dir = BASE_DIR / "outputs"
        outputs_dir.mkdir(exist_ok=True)
        job_info_path = outputs_dir / "mistral_job_info.json"
        
        job_info = {
            "job_id": created_job.id,
            "train_file_id": train_file_id,
            "val_file_id": val_file_id,
            "timestamp": time.time()
        }
        with open(job_info_path, "w") as f:
            json.dump(job_info, f, indent=2)
            
        print(f"\n[INFO] Saved job tracking info to: {job_info_path}")
        print("[INFO] You can monitor the job progress on the Mistral La Plateforme console.")
        
    except Exception as e:
        print(f"\n[ERROR] Failed to trigger fine-tuning job: {e}")

if __name__ == "__main__":
    main()
