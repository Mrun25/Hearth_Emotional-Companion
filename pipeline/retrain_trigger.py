# -*- coding: utf-8 -*-
"""
retrain_trigger.py -- Fumii Automated Retraining on Together AI
===============================================================
Called automatically by eval_runner.py when the model's pass rate drops
below the threshold.  Can also be run manually to force a retrain.

Workflow:
  1. Read fumii_train.jsonl (the clean training dataset)
  2. Upload it to Together AI via the Files API
  3. Start a fine-tuning job on the base model
  4. Poll the job status every 60 seconds until complete or failed
  5. On success → write new model ID to active_model.txt
  6. Log every event with timestamp to fumii_eval_log.txt

Usage (manual trigger):
    python pipeline/retrain_trigger.py

Designed to be called as a subprocess by eval_runner.py; all output goes
to stdout which eval_runner.py redirects to the shared log file.
"""

from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path

# ── Bootstrap ─────────────────────────────────────────────────────────────────
PIPELINE_DIR = Path(__file__).resolve().parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

import config


# ── Logging helper ─────────────────────────────────────────────────────────────

def _log(msg: str) -> None:
    """Print to stdout (eval_runner.py captures this) and append to log file."""
    print(msg, flush=True)
    try:
        with open(config.LOG_FILE, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
    except Exception as e:
        print(f"[LOG ERROR] {e}", file=sys.stderr, flush=True)


def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ── Together AI helpers ────────────────────────────────────────────────────────

def _get_client():
    try:
        from together import Together
    except ImportError:
        raise RuntimeError(
            "The 'together' package is not installed. Run: pip install together"
        )
    if not config.TOGETHER_API_KEY:
        raise RuntimeError(
            "TOGETHER_API_KEY is not set in .env."
        )
    return Together(api_key=config.TOGETHER_API_KEY)


def upload_dataset(client, dataset_path: Path) -> str:
    """
    Upload fumii_train.jsonl to Together AI Files API.
    Returns the file_id string.
    """
    _log(f"[RETRAIN] [{_ts()}] Uploading dataset: {dataset_path}")
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Training dataset not found at {dataset_path}.\n"
            "Please create fumii_train.jsonl in the pipeline/ folder."
        )

    with open(dataset_path, "rb") as f:
        response = client.files.upload(
            file=(dataset_path.name, f, "application/json"),
            purpose="fine-tune",
        )

    file_id = response.id
    _log(f"[RETRAIN] Dataset uploaded. File ID: {file_id}")
    return file_id


def start_finetune_job(client, file_id: str) -> str:
    """
    Start a fine-tuning job on Together AI.
    Returns the job_id string.
    """
    base_model = config.BASE_MODEL
    job_name   = f"fumii-retrain-{datetime.now().strftime('%Y%m%d-%H%M')}"

    _log(f"[RETRAIN] [{_ts()}] Starting fine-tune job:")
    _log(f"          Base model : {base_model}")
    _log(f"          Job name   : {job_name}")
    _log(f"          File ID    : {file_id}")

    response = client.fine_tuning.create(
        model=base_model,
        training_file=file_id,
        n_epochs=3,
        learning_rate=2e-5,
        batch_size=4,
        suffix=job_name,
    )

    job_id = response.id
    _log(f"[RETRAIN] Fine-tune job created. Job ID: {job_id}")
    return job_id


def poll_until_complete(client, job_id: str, poll_interval: int = 60) -> dict:
    """
    Poll the fine-tuning job every `poll_interval` seconds until
    it reaches 'completed' or 'failed'.

    Returns the final job status dict.
    """
    _log(f"[RETRAIN] Polling job {job_id} every {poll_interval}s …")

    terminal_states = {"completed", "failed", "cancelled", "error"}
    elapsed = 0

    while True:
        try:
            job = client.fine_tuning.retrieve(job_id)
            status = job.status
        except Exception as e:
            _log(f"[RETRAIN] [{_ts()}] Poll error (will retry): {e}")
            time.sleep(poll_interval)
            elapsed += poll_interval
            continue

        _log(
            f"[RETRAIN] [{_ts()}]  status={status}  "
            f"elapsed={elapsed // 60}m{elapsed % 60:02d}s"
        )

        if status in terminal_states:
            return {"status": status, "job": job, "elapsed_seconds": elapsed}

        time.sleep(poll_interval)
        elapsed += poll_interval


def save_new_model_id(job) -> str:
    """
    Extract the output model ID from a completed job and write it to
    active_model.txt.  Returns the new model ID.
    """
    # Together AI stores the output model as output_name or fine_tuned_model
    new_model_id = getattr(job, "output_name", None) or getattr(job, "fine_tuned_model", None)

    if not new_model_id:
        raise ValueError(
            f"Could not extract output model ID from completed job: {job}"
        )

    config.ACTIVE_MODEL_FILE.write_text(new_model_id, encoding="utf-8")
    _log(f"[RETRAIN] New model ID saved to {config.ACTIVE_MODEL_FILE}")
    _log(f"[RETRAIN] Active model is now: {new_model_id}")
    return new_model_id


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    _log("\n" + "=" * 65)
    _log(f"  FUMII RETRAIN TRIGGER — {_ts()}")
    _log("=" * 65)

    # Config validation
    warnings = config.validate_config()
    for w in warnings:
        _log(f"[CONFIG WARN] {w}")

    dataset_path = config.TRAIN_DATASET_PATH

    try:
        client = _get_client()

        # Step 1: Upload dataset
        file_id = upload_dataset(client, dataset_path)

        # Step 2: Start fine-tune job
        job_id = start_finetune_job(client, file_id)

        # Step 3: Poll until done
        _log(f"\n[RETRAIN] Job is running. This takes 20–40 minutes on average.")
        _log(f"[RETRAIN] Polling started at {_ts()}")
        result = poll_until_complete(client, job_id, poll_interval=60)

        # Step 4: Handle result
        status = result["status"]
        job    = result["job"]
        mins   = result["elapsed_seconds"] // 60

        if status == "completed":
            new_model_id = save_new_model_id(job)
            _log(f"\n[RETRAIN] ✓ COMPLETED in ~{mins} minutes.")
            _log(f"[RETRAIN]   New model ID : {new_model_id}")
            _log(f"[RETRAIN]   Saved to     : {config.ACTIVE_MODEL_FILE}")
            _log("[RETRAIN] The next eval run will automatically use the new model.")
        else:
            _log(f"\n[RETRAIN] ✗ Job ended with status: {status}")
            _log(f"[RETRAIN]   Job ID  : {job_id}")
            _log(f"[RETRAIN]   Elapsed : {mins} minutes")
            _log("[RETRAIN]   Active model unchanged — check Together AI dashboard.")
            sys.exit(1)

    except FileNotFoundError as e:
        _log(f"\n[ERROR] Dataset not found: {e}")
        _log("[RETRAIN] Aborted — no changes made to active_model.txt")
        sys.exit(1)
    except RuntimeError as e:
        _log(f"\n[ERROR] Configuration error: {e}")
        sys.exit(1)
    except Exception as e:
        _log(f"\n[ERROR] Unexpected error during retraining: {e}")
        import traceback
        _log(traceback.format_exc())
        sys.exit(1)

    _log(f"\n[DONE] retrain_trigger.py finished at {_ts()}\n")


if __name__ == "__main__":
    main()
