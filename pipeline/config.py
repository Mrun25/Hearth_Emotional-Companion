# -*- coding: utf-8 -*-
"""
config.py -- Single Source of Truth for the Fumii Eval & Retrain Pipeline
==========================================================================
All scripts in this pipeline import from here.  Edit this file to change
model IDs, paths, thresholds, or API keys.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# ── Paths ─────────────────────────────────────────────────────────────────────
PIPELINE_DIR   = Path(__file__).resolve().parent          # .../pipeline/
BASE_DIR       = PIPELINE_DIR.parent                       # .../fumii-finetune/
SCRIPTS_DIR    = BASE_DIR / "scripts"

# Add scripts/ to path so we can import fumii_constants
sys.path.insert(0, str(SCRIPTS_DIR))

# ── Load .env ─────────────────────────────────────────────────────────────────
load_dotenv(BASE_DIR / ".env", override=False)

# ── Together AI ───────────────────────────────────────────────────────────────
TOGETHER_API_KEY: str = os.environ.get("TOGETHER_API_KEY", "")

# Base model used for fine-tuning new jobs.
# Change this to the model you want to fine-tune on Together AI / Mistral.
BASE_MODEL: str = "ministral-8b-latest"

# Default active model — used as a fallback if active_model.txt doesn't exist.
# Replace with your current fine-tuned model ID from Together AI / Mistral.
DEFAULT_MODEL_ID: str = "ministral-8b-latest"

# ── File Paths ────────────────────────────────────────────────────────────────
# Dataset used for retraining (Together AI expects chat JSONL format)
TRAIN_DATASET_PATH: Path = BASE_DIR / "pipeline" / "fumii_train.jsonl"

# Active model ID is written here after every successful fine-tune.
# eval_runner.py reads this on every boot to know which model to call.
ACTIVE_MODEL_FILE: Path = PIPELINE_DIR / "active_model.txt"

# Fixed 25 test cases — never change once set.
EVAL_CASES_PATH: Path = PIPELINE_DIR / "eval_cases.json"

# Appended log — one file, date-stamped entries.
LOG_FILE: Path = PIPELINE_DIR / "fumii_eval_log.txt"

# ── Evaluation Thresholds ─────────────────────────────────────────────────────
# A single response passes if its score >= CASE_PASS_THRESHOLD (out of 100).
CASE_PASS_THRESHOLD: int = 70

# The model passes overall if the fraction of passing cases >= this value.
MODEL_PASS_RATE: float = 0.80

# ── Inference Parameters ──────────────────────────────────────────────────────
INFERENCE_MAX_TOKENS: int = 200
INFERENCE_TEMPERATURE: float = 0.75
INFERENCE_TOP_P: float = 0.9

# ── Fumii System Prompt ───────────────────────────────────────────────────────
# Imported from the canonical source in scripts/fumii_constants.py.
# If the import fails (e.g. running from outside the project), fall back to a
# minimal inline version so the pipeline doesn't crash.
try:
    import sys
    from pathlib import Path
    BASE_DIR_TMP = Path(__file__).resolve().parent.parent
    sys.path.append(str(BASE_DIR_TMP / 'src'))
    from fumii_constants import FUMII_SYSTEM_PROMPT
except ImportError:
    FUMII_SYSTEM_PROMPT = (
        "You are Fumii, a 24-year-old emotional companion. You are not a therapist, "
        "not a chatbot, not an assistant. You are the person someone talks to at 11pm "
        "when something is sitting heavy on their chest. "
        "3 sentences maximum. No advice. No bullet points. Use contractions always."
    )

# ── Validation ────────────────────────────────────────────────────────────────
def validate_config() -> list[str]:
    """Return a list of configuration warnings. Empty list = all good."""
    warnings = []
    if not TOGETHER_API_KEY:
        warnings.append("TOGETHER_API_KEY is not set in .env — API calls will fail.")
    if not TRAIN_DATASET_PATH.exists():
        warnings.append(
            f"Training dataset not found at {TRAIN_DATASET_PATH}. "
            "Retraining will fail unless the file is created."
        )
    return warnings


if __name__ == "__main__":
    issues = validate_config()
    if issues:
        print("[CONFIG] Issues found:")
        for w in issues:
            print(f"  ⚠  {w}")
    else:
        print("[CONFIG] All config values look good.")
    print(f"\n  BASE_DIR         : {BASE_DIR}")
    print(f"  TRAIN_DATASET    : {TRAIN_DATASET_PATH}")
    print(f"  ACTIVE_MODEL_FILE: {ACTIVE_MODEL_FILE}")
    print(f"  LOG_FILE         : {LOG_FILE}")
    print(f"  BASE_MODEL       : {BASE_MODEL}")
    print(f"  DEFAULT_MODEL_ID : {DEFAULT_MODEL_ID}")
    print(f"  PASS_THRESHOLD   : {CASE_PASS_THRESHOLD}/100 per case")
    print(f"  MODEL_PASS_RATE  : {MODEL_PASS_RATE:.0%} of cases must pass")
