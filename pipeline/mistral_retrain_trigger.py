# -*- coding: utf-8 -*-
"""
mistral_retrain_trigger.py -- Automated Mistral Retraining Pipeline
====================================================================
Called automatically by eval_runner.py when the model's pass rate drops.
"""

from __future__ import annotations

import os
import sys
import time
import subprocess
from datetime import datetime
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parent
BASE_DIR = PIPELINE_DIR.parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

import config

def _log(msg: str) -> None:
    print(msg, flush=True)
    try:
        with open(config.LOG_FILE, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
    except Exception:
        pass

def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def _get_client():
    try:
        from mistralai.client import Mistral
    except ImportError:
        raise RuntimeError("mistralai package is not installed.")
    
    from dotenv import load_dotenv
    load_dotenv(BASE_DIR / ".env", override=True)
    
    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        raise RuntimeError("MISTRAL_API_KEY is not set in .env.")
    return Mistral(api_key=api_key)

def upload_file(client, filepath: Path) -> str:
    _log(f"[RETRAIN] [{_ts()}] Uploading {filepath.name} to Mistral API...")
    with open(filepath, "rb") as f:
        uploaded_file = client.files.upload(
            file={"file_name": filepath.name, "content": f},
            purpose="fine-tune"
        )
    _log(f"[RETRAIN] Uploaded {filepath.name}. File ID: {uploaded_file.id}")
    return uploaded_file.id

def start_finetune_job(client, train_file_id: str, val_file_id: str = None) -> str:
    _log(f"[RETRAIN] [{_ts()}] Starting Mistral fine-tune job...")
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

    job = client.fine_tuning.jobs.create(**job_kwargs)
    _log(f"[RETRAIN] Fine-tune job created. Job ID: {job.id}")
    return job.id

def poll_until_complete(client, job_id: str, poll_interval: int = 60) -> dict:
    _log(f"[RETRAIN] Polling Mistral job {job_id} every {poll_interval}s...")
    terminal_states = {"SUCCESS", "FAILED", "CANCELLED", "ERROR"}
    elapsed = 0

    while True:
        try:
            job = client.fine_tuning.jobs.get(job_id=job_id)
            status = job.status
        except Exception as e:
            _log(f"[RETRAIN] [{_ts()}] Poll error (will retry): {e}")
            time.sleep(poll_interval)
            elapsed += poll_interval
            continue

        _log(f"[RETRAIN] [{_ts()}]  status={status}  elapsed={elapsed // 60}m{elapsed % 60:02d}s")

        if status in terminal_states:
            return {"status": status, "job": job, "elapsed_seconds": elapsed}

        time.sleep(poll_interval)
        elapsed += poll_interval

def main():
    _log("\n" + "=" * 65)
    _log(f"  FUMII MISTRAL RETRAIN TRIGGER — {_ts()}")
    _log("=" * 65)

    try:
        # Step 1: Data Preparation
        _log(f"[RETRAIN] Generating new data using prepare_data.py...")
        subprocess.run([sys.executable, str(BASE_DIR / "scripts" / "prepare_data.py"), "--generate"], check=True)
        subprocess.run([sys.executable, str(BASE_DIR / "scripts" / "prepare_data.py")], check=True)
        
        train_path = BASE_DIR / "data" / "splits" / "train.jsonl"
        val_path = BASE_DIR / "data" / "splits" / "val.jsonl"
        
        if not train_path.exists():
            raise FileNotFoundError("train.jsonl not found after preparation.")

        client = _get_client()

        # Step 2: Upload dataset
        train_file_id = upload_file(client, train_path)
        val_file_id = upload_file(client, val_path) if val_path.exists() else None

        # Step 3: Start job
        time.sleep(5) # Wait for ingestion
        job_id = start_finetune_job(client, train_file_id, val_file_id)

        # Step 4: Poll
        result = poll_until_complete(client, job_id, poll_interval=60)
        status = result["status"]
        job = result["job"]

        if status == "SUCCESS":
            new_model_id = job.fine_tuned_model
            config.ACTIVE_MODEL_FILE.write_text(new_model_id, encoding="utf-8")
            _log(f"\n[RETRAIN] ✓ COMPLETED.")
            _log(f"[RETRAIN]   New model ID : {new_model_id}")
            
            # Step 5: Re-Evaluate
            _log(f"[RETRAIN] Spawning eval_runner.py to re-evaluate the new model...")
            eval_script = PIPELINE_DIR / "eval_runner.py"
            subprocess.Popen([sys.executable, str(eval_script)], cwd=str(BASE_DIR))
        else:
            _log(f"\n[RETRAIN] ✗ Job ended with status: {status}")
            sys.exit(1)

    except Exception as e:
        _log(f"\n[ERROR] Unexpected error: {e}")
        import traceback
        _log(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()
