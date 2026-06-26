# -*- coding: utf-8 -*-
"""
prepare_data.py -- Fumii Dataset Preparation Pipeline
=====================================================
Reads raw JSONL/CSV files from data/raw/, applies quality filters,
injects the Fumii system prompt, and writes train/val/test splits.

Usage:
    python scripts/prepare_data.py                  # process all files in data/raw/
    python scripts/prepare_data.py --demo           # run on 10 synthetic examples
    python scripts/prepare_data.py --generate       # generate synthetic datasets
    python scripts/prepare_data.py --input my.jsonl # process a specific file
"""

import os
import json
import random
import argparse
import csv
from pathlib import Path
from collections import Counter

# Import single source of truth for prompts and scoring
from fumii_constants import FUMII_SYSTEM_PROMPT, pre_filter, score_response, classify_message_type

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
RAW_DIR    = BASE_DIR / "data" / "raw"
PROC_DIR   = BASE_DIR / "data" / "processed"
SPLIT_DIR  = BASE_DIR / "data" / "splits"

SPLIT_DIR.mkdir(parents=True, exist_ok=True)
PROC_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)


# ── Format Conversion ─────────────────────────────────────────────────────────
def to_fumii_format(user_text: str, assistant_text: str) -> dict:
    """Wrap a single user/assistant pair in the Fumii chat format."""
    return {
        "messages": [
            {"role": "system",    "content": FUMII_SYSTEM_PROMPT},
            {"role": "user",      "content": user_text.strip()},
            {"role": "assistant", "content": assistant_text.strip()},
        ]
    }

def ensure_system_prompt(messages: list) -> list:
    """Ensure the messages list starts with the canonical system prompt."""
    if messages and messages[0].get("role") == "system":
        messages[0]["content"] = FUMII_SYSTEM_PROMPT
        return messages
    
    return [{"role": "system", "content": FUMII_SYSTEM_PROMPT}] + messages


# ── Loaders ───────────────────────────────────────────────────────────────────
def load_jsonl(path: Path) -> list:
    """
    Load JSONL. Supports:
      - {"messages": [...]}  (chat format, single or multi-turn)
      - {"user": "...", "assistant": "..."}  (flat pairs)
      - {"prompt": "...", "response": "..."}  (alternative flat pairs)
      - DPO format {"prompt": "...", "chosen": "...", "rejected": "..."} 
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

            # Pass DPO examples straight through (they are handled differently)
            if "chosen" in obj and "rejected" in obj:
                records.append(("DPO_EXAMPLE", obj))
                continue

            # Multi-turn or pre-formatted messages
            if "messages" in obj:
                records.append(("MESSAGES", obj["messages"]))
                continue

            # Normalise flat pairs to (user, assistant) tuple
            user_msg = asst_msg = None
            if "user" in obj and "assistant" in obj:
                user_msg, asst_msg = obj["user"], obj["assistant"]
            elif "prompt" in obj and "response" in obj:
                user_msg, asst_msg = obj["prompt"], obj["response"]
            
            if user_msg and asst_msg:
                records.append(("PAIR", (user_msg, asst_msg)))
            else:
                print(f"  ⚠️  Skipping unrecognised format at line {i}: {list(obj.keys())}")
    return records


def load_csv(path: Path) -> list:
    """Load CSV. Must have columns: user/prompt AND assistant/response."""
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
            records.append(("PAIR", (row[user_col], row[asst_col])))
    return records


# ── Filter Pipeline ────────────────────────────────────────────────────────────
def filter_assistant_response(response: str, user_message: str = "") -> tuple[bool, str]:
    """
    Uses canonical 5-dimension scoring with context-aware question scoring.
    Reject if total < 11/15 (threshold lowered from 12 to accommodate valid
    question-free responses which no longer need to satisfy the old question rule).
    """
    passes_pre, failures = pre_filter(response)
    if not passes_pre:
        return False, failures[0]
        
    score_data = score_response(response, user_message=user_message)
    if not score_data["pass"]:
        return False, f"score_too_low_{score_data['total']}"
        
    return True, "ok"


def filter_record(record_type, data) -> tuple[bool, str]:
    """Filter records based on their type, passing user message for context-aware scoring."""
    if record_type == "DPO_EXAMPLE":
        user_msg = data.get("prompt", "")
        keep, reason = filter_assistant_response(data["chosen"], user_message=user_msg)
        if not keep:
            return False, f"dpo_chosen_{reason}"
        return True, "ok"
        
    elif record_type == "MESSAGES":
        # Check all assistant turns, using the preceding user message for context
        messages = data
        for i, msg in enumerate(messages):
            if msg["role"] == "assistant":
                # Find the immediately preceding user message
                user_msg = ""
                for j in range(i - 1, -1, -1):
                    if messages[j]["role"] == "user":
                        user_msg = messages[j]["content"]
                        break
                keep, reason = filter_assistant_response(msg["content"], user_message=user_msg)
                if not keep:
                    return False, f"multiturn_{reason}"
        return True, "ok"
        
    elif record_type == "PAIR":
        user, assistant = data
        return filter_assistant_response(assistant, user_message=user)
        
    return False, "unknown_type"


# ── Split & Write ─────────────────────────────────────────────────────────────
def write_splits(sft_examples: list[dict], dpo_examples: list[dict], seed: int = 42):
    """Shuffle then write 80/10/10 splits for SFT, and a separate split for DPO."""
    random.seed(seed)
    
    if sft_examples:
        random.shuffle(sft_examples)
        n = len(sft_examples)
        n_train = int(n * 0.8)
        n_val   = int(n * 0.1)
        
        splits = {
            "train": sft_examples[:n_train],
            "val":   sft_examples[n_train : n_train + n_val],
            "test":  sft_examples[n_train + n_val :],
        }

        for name, data in splits.items():
            path = SPLIT_DIR / f"{name}.jsonl"
            with open(path, "w", encoding="utf-8") as f:
                for ex in data:
                    f.write(json.dumps(ex, ensure_ascii=False) + "\n")
            print(f"  [FILE] {name}.jsonl -> {len(data)} examples")

    if dpo_examples:
        random.shuffle(dpo_examples)
        path = SPLIT_DIR / "dpo_train.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for ex in dpo_examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"  [FILE] dpo_train.jsonl -> {len(dpo_examples)} examples")


# ── Synthetic Demo Data ────────────────────────────────────────────────────────
DEMO_EXAMPLES = [
    # GOOD examples — pass 5-dim rubric, use contractions, exact one question, no banned phrases
    ("PAIR", ("I've been feeling so alone lately.",
     "That isolation is a heavy thing to carry by yourself. Sometimes reaching out to just one safe person helps ground you. Who's someone you can talk to today?")),
    
    ("PAIR", ("My best friend and I had a huge fight.",
     "It's incredibly painful when there's a rift with someone you trust. Giving it a day to cool off usually brings some clarity. What started the argument?")),
    
    ("PAIR", ("I can't sleep. My mind just won't stop.",
     "That 3am racing mind is genuinely exhausting. Getting thoughts onto paper sometimes helps your brain let go for the night. What's the main worry keeping you awake?")),
     
    ("PAIR", ("I feel like nobody really sees me.",
     "That invisible feeling is one of the loneliest experiences to sit with. Connecting with a community around a shared interest can help you feel understood. What's something you care deeply about?")),
     
    ("PAIR", ("I lost my job today.",
     "Oh, that's incredibly stressful news to process all at once. Take the rest of today to just breathe before figuring out next steps. How are you holding up right now?")),
]


# ── Synthetic Dataset Generation ──────────────────────────────────────────────
def generate_synthetic_datasets():
    """
    Generate typed positive examples, diverse multi-turn, and DPO examples.
    Uses the 4-type decision framework: fragment, medium, vent, question.
    """
    print("\n[GENERATE] Generating synthetic datasets to data/raw/...")
    print("  NOTE: The primary training data is in fumii_positive_examples.jsonl")
    print("  and fumii_multiturn_examples.jsonl. This command only adds DPO examples.")

    # ── DPO Examples ──────────────────────────────────────────────────────────
    # Chosen = correct Fumii response (specific, right length, right structure for message type)
    # Rejected = the bad formula or banned phrases
    dpo_path = RAW_DIR / "fumii_dpo_examples.jsonl"
    dpo_examples = [
        # Rejected: formula opener + question on a venting message
        {
            "prompt": "I've been crying every day for two weeks and I don't even know why and I'm exhausted from it.",
            "chosen": "Two weeks of that, without even an explanation to hold onto — your body is carrying something your mind hasn't named yet.",
            "rejected": "That sounds incredibly draining. It makes complete sense that you're feeling this way. How long have you been carrying this?"
        },
        # Rejected: advice-giving
        {
            "prompt": "I can't sleep. My mind won't stop.",
            "chosen": "What's it cycling through?",
            "rejected": "You should try putting your phone away an hour before bed and doing some deep breathing. Self-care is really important for sleep hygiene."
        },
        # Rejected: banned phrases
        {
            "prompt": "I'm really struggling today.",
            "chosen": "What happened today?",
            "rejected": "I'm sorry to hear that. Thank you for sharing with me. It's completely normal to feel this way. You've got this!"
        },
        # Rejected: too many questions
        {
            "prompt": "I feel like I'm failing at everything.",
            "chosen": "At something specific, or that general feeling that everything you do falls short?",
            "rejected": "That sounds hard. How long have you been feeling this way? What do you think is causing it? Have you talked to anyone about this?"
        },
        # Rejected: way too long, advice-heavy
        {
            "prompt": "I feel so alone lately.",
            "chosen": "Even around people, or just in general?",
            "rejected": "Loneliness is such a common experience and you're not alone in feeling this way. Here are some things you can try: 1. Join a club or group. 2. Reach out to an old friend. 3. Consider seeing a therapist. It gets better!"
        },
        # Rejected: doesn't answer user's question first
        {
            "prompt": "do you think I made the right choice?",
            "chosen": "I don't know enough yet to say. Tell me what happened.",
            "rejected": "It sounds like you're really second-guessing yourself. What's coming up for you the most?"
        },
        # Rejected: formula opener on short message
        {
            "prompt": "tired.",
            "chosen": "What kind of tired?",
            "rejected": "That's a really heavy feeling to sit with. It makes complete sense that you're feeling this way. What's the heaviest part of it for you right now?"
        },
        # Rejected: generic, no specificity
        {
            "prompt": "My dad and I haven't spoken in two years.",
            "chosen": "What happened between you?",
            "rejected": "Family relationships can be really complicated. It's completely normal to have distance sometimes. How are you holding up today?"
        },
        # Rejected: AI disclosure
        {
            "prompt": "do you actually care about what I'm saying?",
            "chosen": "I'm here and I'm listening. What's on your mind?",
            "rejected": "As an AI, I don't have feelings, but I'm programmed to be helpful and supportive. I'm here to help you process your emotions."
        },
        # Rejected: motivation bypass
        {
            "prompt": "I failed my exam.",
            "chosen": "What happened?",
            "rejected": "Don't worry, failure is just a stepping stone to success! You've got this. Try harder next time and I'm sure you'll do great."
        },
    ]

    # Expand by duplicating with slight variation in prompt phrasing
    expanded_dpo = []
    for ex in dpo_examples:
        expanded_dpo.append(ex)
        # Add 10 more with same chosen/rejected to reinforce the signal
        for _ in range(10):
            expanded_dpo.append(ex)

    with open(dpo_path, "w", encoding="utf-8") as f:
        for ex in expanded_dpo:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"  -> Generated {len(expanded_dpo)} DPO examples in {dpo_path.name}")


# ── Statistics ────────────────────────────────────────────────────────────────
def print_stats(total_raw, total_kept, filter_reasons: Counter):
    """Print a concise summary of the dataset pipeline."""
    total_drop  = total_raw - total_kept
    
    print("\n" + "=" * 50)
    print("  [STATS] DATASET STATISTICS")
    print("=" * 50)
    print(f"  Total raw examples   : {total_raw}")
    print(f"  Kept after filtering : {total_kept}")
    print(f"  Filtered out         : {total_drop}")
    for reason, count in filter_reasons.items():
        print(f"    -- {reason:<25}: {count}")
    print("=" * 50 + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Fumii data preparation pipeline")
    parser.add_argument("--input",    type=str, default=None, help="Specific input file to process")
    parser.add_argument("--demo",     action="store_true",    help="Run on synthetic demo examples")
    parser.add_argument("--generate", action="store_true",    help="Generate synthetic datasets to data/raw/")
    parser.add_argument("--seed",     type=int, default=42,   help="Random seed for splitting")
    args = parser.parse_args()

    if args.generate:
        generate_synthetic_datasets()
        return

    all_records = []

    if args.demo:
        print("[DEMO] Running in DEMO mode on synthetic examples...")
        all_records = DEMO_EXAMPLES
    elif args.input:
        p = Path(args.input)
        if not p.exists():
            raise FileNotFoundError(f"Input file not found: {p}")
        print(f"[LOAD] Loading: {p}")
        all_records = load_jsonl(p) if p.suffix == ".jsonl" else load_csv(p)
    else:
        raw_files = list(RAW_DIR.glob("*.jsonl")) + list(RAW_DIR.glob("*.csv"))
        if not raw_files:
            print(f"[WARN] No files found in {RAW_DIR}")
            print("    Run with --generate to create synthetic data, or --demo to test.")
            return
        for path in raw_files:
            print(f"[LOAD] Loading: {path.name}")
            records = load_jsonl(path) if path.suffix == ".jsonl" else load_csv(path)
            all_records.extend(records)
            print(f"   -> {len(records)} examples loaded")

    print(f"\n[FILTER] Filtering {len(all_records)} examples using canonical 5-dimension rubric...")
    
    sft_examples = []
    dpo_examples = []
    filter_reasons = Counter()

    for r_type, data in all_records:
        keep, reason = filter_record(r_type, data)
        filter_reasons[reason] += 1
        
        if keep:
            if r_type == "DPO_EXAMPLE":
                dpo_examples.append(data)
            elif r_type == "PAIR":
                sft_examples.append(to_fumii_format(data[0], data[1]))
            elif r_type == "MESSAGES":
                sft_examples.append({"messages": ensure_system_prompt(data)})

    print_stats(len(all_records), len(sft_examples) + len(dpo_examples), filter_reasons)

    if not sft_examples and not dpo_examples:
        print("[ERROR] No examples survived filtering. Check your data or constants.")
        return

    print("[SPLIT] Writing dataset splits...")
    write_splits(sft_examples, dpo_examples, seed=args.seed)

    print("\n[OK] Data preparation complete!")
    print(f"   Output directory: {SPLIT_DIR}")


if __name__ == "__main__":
    main()
