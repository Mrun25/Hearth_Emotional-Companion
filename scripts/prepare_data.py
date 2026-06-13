# -*- coding: utf-8 -*-
"""
prepare_data.py -- Fumii Dataset Preparation Pipeline
=====================================================
Reads raw JSONL/CSV files from data/raw/, applies quality filters,
injects the Fumii system prompt, and writes train/val/test splits.

Usage:
    python scripts/prepare_data.py                  # process all files in data/raw/
    python scripts/prepare_data.py --demo           # run on 10 synthetic examples (Step 2 ✅ test)
    python scripts/prepare_data.py --input my.jsonl # process a specific file
"""

import os
import re
import json
import random
import argparse
import csv
from pathlib import Path
from collections import Counter

# ── Paths ────────────────────────────────────────────────────────────────────
# WHY resolve from __file__: script works regardless of cwd
BASE_DIR   = Path(__file__).resolve().parent.parent
RAW_DIR    = BASE_DIR / "data" / "raw"
PROC_DIR   = BASE_DIR / "data" / "processed"
SPLIT_DIR  = BASE_DIR / "data" / "splits"

SPLIT_DIR.mkdir(parents=True, exist_ok=True)
PROC_DIR.mkdir(parents=True, exist_ok=True)

# ── Fumii System Prompt ───────────────────────────────────────────────────────
# WHY this exact wording: it encodes persona, hard constraints (≤3 sentences,
# open-ended question), and tone without being preachy.
FUMII_SYSTEM_PROMPT = (
    "You are Fumii — a warm, calm, wise, and playful emotional companion. "
    "You are NOT a therapist. NEVER respond with more than 3 sentences. "
    "ALWAYS ask one open-ended follow-up question. NEVER give unsolicited advice. "
    "Speak like a thoughtful 24-year-old with deep emotional wisdom. "
    "Be curious. Be present. Be real."
)

# ── Anti-Pattern Phrases ──────────────────────────────────────────────────────
# WHY lowercase match: catches "Life is about" and "life is about" equally.
ANTI_PATTERNS = [
    "life is about",
    "everything happens for a reason",
    "you should try",
    "as an ai",
    "i recommend",
    "have you considered",
    "it's important to",
]

# ── Sentence Splitter ─────────────────────────────────────────────────────────
SENTENCE_END = re.compile(r'(?<=[.!?])\s+')

def count_sentences(text: str) -> int:
    """Split on sentence-ending punctuation and count fragments."""
    # Strip trailing whitespace then split; filter empty fragments
    parts = SENTENCE_END.split(text.strip())
    return len([p for p in parts if p.strip()])

def has_anti_pattern(text: str) -> bool:
    """Return True if any banned phrase is found (case-insensitive)."""
    lower = text.lower()
    return any(phrase in lower for phrase in ANTI_PATTERNS)

# ── Format Conversion ─────────────────────────────────────────────────────────
def to_fumii_format(user_text: str, assistant_text: str) -> dict:
    """
    Wrap a user/assistant pair in the Fumii chat format.
    WHY always inject system prompt here: ensures EVERY training example
    has the persona baked in, not just some.
    """
    return {
        "messages": [
            {"role": "system",    "content": FUMII_SYSTEM_PROMPT},
            {"role": "user",      "content": user_text.strip()},
            {"role": "assistant", "content": assistant_text.strip()},
        ]
    }

# ── Loaders ───────────────────────────────────────────────────────────────────
def load_jsonl(path: Path) -> list[dict]:
    """
    Load JSONL. Expected formats:
      - {"messages": [...]}  (already in chat format)
      - {"user": "...", "assistant": "..."}  (flat pairs)
      - {"prompt": "...", "response": "..."}  (alternative flat pairs)
    """
    records = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  ⚠️  Skipping malformed JSON at line {i}: {e}")
                continue

            # Normalise to (user, assistant) tuple
            if "messages" in obj:
                msgs = obj["messages"]
                user_msg  = next((m["content"] for m in msgs if m["role"] == "user"), None)
                asst_msg  = next((m["content"] for m in msgs if m["role"] == "assistant"), None)
            elif "user" in obj and "assistant" in obj:
                user_msg, asst_msg = obj["user"], obj["assistant"]
            elif "prompt" in obj and "response" in obj:
                user_msg, asst_msg = obj["prompt"], obj["response"]
            else:
                print(f"  ⚠️  Skipping unrecognised format at line {i}: {list(obj.keys())}")
                continue

            if user_msg and asst_msg:
                records.append((user_msg, asst_msg))
    return records


def load_csv(path: Path) -> list[tuple]:
    """
    Load CSV. Must have columns: user/prompt AND assistant/response.
    """
    records = []
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, 1):
            user_col = next((k for k in row if k.lower() in ("user", "prompt")), None)
            asst_col = next((k for k in row if k.lower() in ("assistant", "response")), None)
            if not user_col or not asst_col:
                if i == 1:
                    print(f"  ⚠️  CSV missing expected columns. Found: {list(row.keys())}")
                continue
            records.append((row[user_col], row[asst_col]))
    return records

# ── Synthetic Demo Data ────────────────────────────────────────────────────────
DEMO_EXAMPLES = [
    # GOOD examples — should pass all filters
    ("I've been feeling so alone lately.",
     "That loneliness sounds really heavy. What does it feel like in your body when it hits?"),
    ("My best friend and I had a huge fight.",
     "Falling out with someone close is really painful. What started the argument, if you don't mind sharing?"),
    ("I can't sleep. My mind just won't stop.",
     "That racing mind at 3am is exhausting. What thoughts keep coming back to you?"),
    ("I feel like nobody really sees me.",
     "That invisible feeling is one of the loneliest things. When was the last time you felt truly seen?"),
    ("I lost my job today.",
     "Oh, that's a lot to absorb all at once. How are you holding up right now?"),
    ("I don't know what I want from life.",
     "That uncertainty can feel unmooring. What used to excite you that maybe doesn't anymore?"),
    ("My anxiety has been really bad this week.",
     "Anxiety at that level is genuinely exhausting. What's been triggering it most this week?"),
    # BAD examples — should be filtered out
    ("I feel really stuck.",
     "Life is about finding your path and everything happens for a reason! You should try journaling."),
    ("I'm so sad.",
     "As an AI, I recommend you seek professional help. It's important to take care of yourself."),
    ("I feel lost.",
     "Have you considered talking to a therapist? They can really help. Also, remember to exercise. "
     "Eat well too. Sleep is important. Build a routine. Focus on the positives. "
     "Gratitude journaling helps. Try meditation. Call a friend."),  # > 3 sentences
]

# ── Filter Pipeline ────────────────────────────────────────────────────────────
def filter_example(user: str, assistant: str) -> tuple[bool, str]:
    """
    Returns (keep: bool, reason: str).
    WHY two separate checks: lets statistics show WHICH filter removed what.
    """
    if count_sentences(assistant) > 3:
        return False, "too_long"
    if has_anti_pattern(assistant):
        return False, "anti_pattern"
    return True, "ok"

# ── Split & Write ─────────────────────────────────────────────────────────────
def write_splits(examples: list[dict], seed: int = 42):
    """Shuffle then write 80/10/10 splits."""
    random.seed(seed)
    random.shuffle(examples)

    n = len(examples)
    n_train = int(n * 0.8)
    n_val   = int(n * 0.1)
    # test gets the remainder (handles rounding)
    splits = {
        "train": examples[:n_train],
        "val":   examples[n_train : n_train + n_val],
        "test":  examples[n_train + n_val :],
    }

    for name, data in splits.items():
        path = SPLIT_DIR / f"{name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for ex in data:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"  [FILE] {name}.jsonl -> {len(data)} examples")

    return splits

# ── Statistics ────────────────────────────────────────────────────────────────
def print_stats(all_pairs: list, filtered_pairs: list, filter_reasons: Counter):
    """Print a concise summary of the dataset pipeline."""
    total_raw   = len(all_pairs)
    total_kept  = len(filtered_pairs)
    total_drop  = total_raw - total_kept

    if total_kept == 0:
        avg_len = 0.0
    else:
        avg_len = sum(
            len(asst.split()) for _, asst in filtered_pairs
        ) / total_kept

    print("\n" + "=" * 50)
    print("  [STATS] DATASET STATISTICS")
    print("=" * 50)
    print(f"  Total raw examples   : {total_raw}")
    print(f"  Kept after filtering : {total_kept}")
    print(f"  Filtered out         : {total_drop}")
    for reason, count in filter_reasons.items():
        print(f"    -- {reason:<15}: {count}")
    print(f"  Avg response length  : {avg_len:.1f} words")
    print("=" * 50 + "\n")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Fumii data preparation pipeline")
    parser.add_argument("--input",  type=str, default=None, help="Specific input file to process")
    parser.add_argument("--demo",   action="store_true",     help="Run on 10 synthetic demo examples")
    parser.add_argument("--seed",   type=int, default=42,    help="Random seed for splitting")
    args = parser.parse_args()

    all_pairs: list[tuple[str, str]] = []

    if args.demo:
        print("[DEMO] Running in DEMO mode on 10 synthetic examples...")
        all_pairs = DEMO_EXAMPLES
    elif args.input:
        p = Path(args.input)
        if not p.exists():
            raise FileNotFoundError(f"Input file not found: {p}")
        print(f"[LOAD] Loading: {p}")
        all_pairs = load_jsonl(p) if p.suffix == ".jsonl" else load_csv(p)
    else:
        # Auto-discover all JSONL and CSV files in data/raw/
        raw_files = list(RAW_DIR.glob("*.jsonl")) + list(RAW_DIR.glob("*.csv"))
        if not raw_files:
            print(f"[WARN] No .jsonl or .csv files found in {RAW_DIR}")
            print("    Run with --demo to test on synthetic data.")
            return
        for path in raw_files:
            print(f"[LOAD] Loading: {path.name}")
            pairs = load_jsonl(path) if path.suffix == ".jsonl" else load_csv(path)
            all_pairs.extend(pairs)
            print(f"   -> {len(pairs)} examples loaded")

    print(f"\n[FILTER] Filtering {len(all_pairs)} examples...")
    filtered_pairs: list[tuple[str, str]] = []
    filter_reasons: Counter = Counter()

    for user, assistant in all_pairs:
        keep, reason = filter_example(user, assistant)
        filter_reasons[reason] += 1
        if keep:
            filtered_pairs.append((user, assistant))

    # Convert to Fumii chat format
    examples = [to_fumii_format(u, a) for u, a in filtered_pairs]

    print_stats(all_pairs, filtered_pairs, filter_reasons)

    if not examples:
        print("[ERROR] No examples survived filtering. Check your data or filters.")
        return

    print("[SPLIT] Writing train/val/test splits...")
    write_splits(examples, seed=args.seed)

    print("\n[OK] Data preparation complete!")
    print(f"   Output directory: {SPLIT_DIR}")


if __name__ == "__main__":
    main()
