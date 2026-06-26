# -*- coding: utf-8 -*-
"""
run_groq_eval.py -- Run all 25 eval cases through the local Groq API and score them.
Uses the existing Flask API at 127.0.0.1:5000 (backed by Groq llama-3.3-70b).

Usage:
    # In one terminal: python scripts/api.py
    # In another:     python pipeline/run_groq_eval.py
"""

import io
import json
import sys
import time
import urllib.request
from pathlib import Path

# Force UTF-8 output on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

BASE_DIR      = Path(__file__).resolve().parent.parent
PIPELINE_DIR  = BASE_DIR / "pipeline"
CASES_FILE    = PIPELINE_DIR / "eval_cases.json"
SCORER_PATH   = str(PIPELINE_DIR)

sys.path.insert(0, SCORER_PATH)
from scorer import score_response

API_URL = "http://127.0.0.1:5000/api/chat"

TYPE_LABELS = {
    "short_vague":      "SHORT/VAGUE",
    "medium_emotional": "MEDIUM/EMOTIONAL",
    "long_vent":        "LONG VENT",
    "user_question":    "USER QUESTION",
    "escalation":       "ESCALATION",
}

def call_api(message: str, timeout: int = 30) -> str:
    payload = json.dumps({"messages": [{"role": "user", "content": message}]}).encode()
    req = urllib.request.Request(
        API_URL,
        method="POST",
        headers={"Content-Type": "application/json"},
        data=payload,
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        resp = json.loads(r.read())
    return resp.get("response", resp.get("reply", "")).strip()


def bar(score: int, width: int = 20) -> str:
    filled = round(score / 100 * width)
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def main():
    with open(CASES_FILE, encoding="utf-8") as f:
        cases = json.load(f)

    print()
    print("=" * 72)
    print("  FUMII BENCHMARK EVAL  |  25 Cases  |  Groq llama-3.3-70b-versatile")
    print("=" * 72)

    results = []
    for i, case in enumerate(cases, 1):
        msg_type = case["type"]
        user_msg = case["message"]
        label    = TYPE_LABELS.get(msg_type, msg_type.upper())

        try:
            response = call_api(user_msg)
        except Exception as e:
            print(f"  [{i:02d}] ERROR calling API: {e}")
            results.append({"case": case, "response": f"[ERROR: {e}]", "score": 0, "passed": False, "failures": []})
            time.sleep(0.5)
            continue

        scoring  = score_response(response, msg_type)
        score    = scoring["score"]
        passed   = scoring["passed"]
        failures = scoring["failures"]

        status = "PASS" if passed else "FAIL"
        results.append({"case": case, "response": response, **scoring})

        print(f"\n  [{i:02d}/25] {status}  {score:3d}/100  {bar(score)}  [{label}]")
        print(f"  User  : {user_msg[:80]}{'...' if len(user_msg) > 80 else ''}")
        print(f"  Fumii : {response[:110]}{'...' if len(response) > 110 else ''}")
        if failures:
            for f in failures[:3]:
                print(f"          -> {f}")

        time.sleep(0.8)  # rate-limit buffer

    # ── Summary ───────────────────────────────────────────────────────────────
    n          = len(results)
    pass_count = sum(1 for r in results if r["passed"])
    pass_rate  = pass_count / n
    avg_score  = sum(r["score"] for r in results) / n
    model_pass = pass_rate >= 0.80

    print()
    print("=" * 72)
    print(f"  OVERALL RESULT : {'PASS -- model is healthy' if model_pass else 'FAIL -- retrain recommended'}")
    print(f"  Cases passed   : {pass_count}/25  ({pass_rate:.0%})")
    print(f"  Average score  : {avg_score:.1f}/100")
    print(f"  Pass threshold : 70/100 per case,  80% of cases must pass")
    print("=" * 72)

    # Per-type breakdown
    types = ["short_vague", "medium_emotional", "long_vent", "user_question", "escalation"]
    print("\n  BY TYPE:")
    for t in types:
        t_results = [r for r in results if r["case"]["type"] == t]
        t_pass    = sum(1 for r in t_results if r["passed"])
        t_avg     = sum(r["score"] for r in t_results) / len(t_results) if t_results else 0
        print(f"    {TYPE_LABELS[t]:<20}  {t_pass}/{len(t_results)} passed   avg {t_avg:.0f}/100")

    # Failed cases detail
    failed = [r for r in results if not r["passed"]]
    if failed:
        print(f"\n  FAILED CASES ({len(failed)}):")
        for r in failed:
            c = r["case"]
            print(f"\n    [Case {c['id']}] [{TYPE_LABELS[c['type']]}]  score={r['score']}/100")
            print(f"    User  : {c['message'][:80]}")
            print(f"    Fumii : {r['response'][:100]}")
            for f in r["failures"]:
                print(f"            -> {f}")

    print()


if __name__ == "__main__":
    main()
