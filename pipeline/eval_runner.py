# -*- coding: utf-8 -*-
"""
eval_runner.py -- Fumii Automated Evaluation Runner
====================================================
Orchestrates the daily evaluation loop:
  1. Load 25 fixed test cases from eval_cases.json
  2. Call the active Fumii model on Together AI for each case
  3. Score every response with scorer.py (100-pt rubric)
  4. Print a summary table to the terminal
  5. Append timestamped results to fumii_eval_log.txt
  6. If pass rate < 80%, automatically call retrain_trigger.py

Run manually:
    python pipeline/eval_runner.py

Or via startup_runner.bat (runs on every Windows boot).
"""

from __future__ import annotations

import json
import sys
import subprocess
from datetime import datetime
from pathlib import Path

# ── Bootstrap: ensure pipeline/ is on sys.path ────────────────────────────────
PIPELINE_DIR = Path(__file__).resolve().parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

import config
from scorer import score_response


# ── Logging helper ─────────────────────────────────────────────────────────────

def _log(msg: str, also_print: bool = True) -> None:
    """Append a line to fumii_eval_log.txt and optionally print to terminal."""
    line = msg + "\n"
    try:
        with open(config.LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line)
    except Exception as e:
        print(f"[LOG ERROR] Could not write to log: {e}", file=sys.stderr)
    if also_print:
        print(msg)


# ── Active model loader ────────────────────────────────────────────────────────

def load_active_model() -> str:
    """
    Read the current model ID from active_model.txt.
    Falls back to config.DEFAULT_MODEL_ID if the file doesn't exist.
    """
    if config.ACTIVE_MODEL_FILE.exists():
        model_id = config.ACTIVE_MODEL_FILE.read_text(encoding="utf-8").strip()
        if model_id:
            return model_id
    _log(
        f"[WARN] {config.ACTIVE_MODEL_FILE} not found or empty — "
        f"falling back to default: {config.DEFAULT_MODEL_ID}"
    )
    return config.DEFAULT_MODEL_ID


# ── Mistral AI inference ──────────────────────────────────────────────────────

def call_mistral(model_id: str, user_message: str) -> str:
    """
    Send a single user message to the active Fumii model on Mistral.
    Returns the assistant's response text, or raises on error.
    """
    try:
        from mistralai.client import Mistral
    except ImportError:
        raise RuntimeError(
            "The 'mistralai' package is not installed. "
            "Run: pip install mistralai"
        )

    import os
    mistral_key = os.environ.get("MISTRAL_API_KEY")
    if not mistral_key:
        raise RuntimeError(
            "MISTRAL_API_KEY is not set in .env. "
            "Please add it before running eval."
        )

    client = Mistral(api_key=mistral_key)

    messages = [
        {"role": "system",  "content": config.FUMII_SYSTEM_PROMPT},
        {"role": "user",    "content": user_message},
    ]

    response = client.chat.complete(
        model=model_id,
        messages=messages,
        temperature=config.INFERENCE_TEMPERATURE,
        max_tokens=config.INFERENCE_MAX_TOKENS,
        top_p=config.INFERENCE_TOP_P,
    )
    return response.choices[0].message.content.strip()


# ── Evaluation logic ───────────────────────────────────────────────────────────

def run_evaluation() -> dict:
    """
    Run the full 25-case evaluation.

    Returns a results dict:
        {
          "model_id":   str,
          "timestamp":  str,
          "cases":      list[dict],   # one per test case
          "pass_count": int,
          "pass_rate":  float,
          "passed":     bool,         # True if >= MODEL_PASS_RATE
        }
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _log("\n" + "=" * 65)
    _log(f"  FUMII EVAL — {timestamp}")
    _log("=" * 65)

    # Load test cases
    with open(config.EVAL_CASES_PATH, encoding="utf-8") as f:
        cases: list[dict] = json.load(f)

    model_id = load_active_model()
    _log(f"  Model : {model_id}")
    _log(f"  Cases : {len(cases)}\n")

    results_cases: list[dict] = []
    errors: int = 0

    for i, case in enumerate(cases, 1):
        user_msg  = case["message"]
        msg_type  = case["type"]
        case_id   = case["id"]

        # Call Mistral AI
        try:
            response = call_mistral(model_id, user_message=user_msg)
        except Exception as e:
            err_msg = f"[ERROR] Case {case_id} — API call failed: {e}"
            _log(err_msg)
            errors += 1
            results_cases.append({
                "id":       case_id,
                "type":     msg_type,
                "message":  user_msg,
                "response": f"[API ERROR: {e}]",
                "score":    0,
                "passed":   False,
                "failures": [err_msg],
            })
            continue

        # Score response
        scoring = score_response(response, msg_type)
        results_cases.append({
            "id":       case_id,
            "type":     msg_type,
            "message":  user_msg,
            "response": response,
            "score":    scoring["score"],
            "passed":   scoring["passed"],
            "failures": scoring["failures"],
            "details":  scoring["details"],
        })

        status = "✓ PASS" if scoring["passed"] else "✗ FAIL"
        _log(
            f"  [{i:02d}/{len(cases)}] {status}  [{scoring['score']:3d}/100]  "
            f"[{msg_type:<18}]  {user_msg[:55]}{'…' if len(user_msg) > 55 else ''}"
        )

    # Summary
    pass_count = sum(1 for r in results_cases if r["passed"])
    pass_rate  = pass_count / len(cases) if cases else 0.0
    passed     = pass_rate >= config.MODEL_PASS_RATE

    _log("\n" + "-" * 65)
    _log(f"  RESULT : {'PASS ✓' if passed else 'FAIL ✗'}")
    _log(f"  Score  : {pass_count}/{len(cases)} cases passed ({pass_rate:.1%})")
    _log(f"  Errors : {errors} API call(s) failed")
    _log("-" * 65)

    # Print detail on failures
    failed_cases = [r for r in results_cases if not r["passed"]]
    if failed_cases:
        _log("\n  FAILED CASES:")
        for r in failed_cases:
            _log(f"\n  ✗ Case {r['id']} [{r['type']}]")
            _log(f"    User    : {r['message'][:80]}")
            _log(f"    Fumii   : {r['response'][:120]}")
            if r.get("failures"):
                for f in r["failures"][:4]:
                    _log(f"    ↳ {f}")

    return {
        "model_id":   model_id,
        "timestamp":  timestamp,
        "cases":      results_cases,
        "pass_count": pass_count,
        "pass_rate":  pass_rate,
        "passed":     passed,
    }


# ── Retrain trigger ────────────────────────────────────────────────────────────

def trigger_retrain() -> None:
    """
    Spawn mistral_retrain_trigger.py as a subprocess so it can run its long polling
    loop without blocking the caller.  Output streams to the shared log.
    """
    retrain_script = PIPELINE_DIR / "mistral_retrain_trigger.py"
    if not retrain_script.exists():
        _log(f"[ERROR] mistral_retrain_trigger.py not found at {retrain_script}")
        return

    _log("\n[RETRAIN] Pass rate below threshold — launching mistral_retrain_trigger.py …")
    try:
        # Run in the background; stdout/stderr go to the log file
        with open(config.LOG_FILE, "a", encoding="utf-8") as log_fh:
            subprocess.Popen(
                [sys.executable, str(retrain_script)],
                stdout=log_fh,
                stderr=log_fh,
                cwd=str(PIPELINE_DIR.parent),
            )
        _log("[RETRAIN] mistral_retrain_trigger.py launched — check log for progress.")
    except Exception as e:
        _log(f"[ERROR] Failed to launch mistral_retrain_trigger.py: {e}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    # Validate config first
    warnings = config.validate_config()
    if warnings:
        for w in warnings:
            _log(f"[CONFIG WARN] {w}")

    try:
        results = run_evaluation()
    except Exception as e:
        _log(f"\n[FATAL] Evaluation crashed: {e}")
        import traceback
        _log(traceback.format_exc())
        sys.exit(1)

    if not results["passed"]:
        trigger_retrain()
    else:
        _log("\n[OK] Model is healthy — no retraining needed.")

    _log("\n[DONE] eval_runner.py finished.\n")


if __name__ == "__main__":
    main()
