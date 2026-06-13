# -*- coding: utf-8 -*-
"""
evaluate.py -- Fumii Fine-Tuned Model Evaluation
=================================================
Loads the Mistral-7B base model + LoRA adapter from outputs/checkpoints/,
runs inference on data/splits/test.jsonl, and scores each response on:

  1. Length Check    -- response <= 3 sentences (pass/fail)
  2. Question Check  -- response contains '?' (proxy for open-ended follow-up)
  3. Anti-Pattern    -- response contains any banned phrase (fail if found)

Outputs a summary pass-rate table and 5 sample model outputs.

Usage:
    python scripts/evaluate.py                          # full evaluation
    python scripts/evaluate.py --demo                   # evaluate on demo splits (no GPU needed for scoring)
    python scripts/evaluate.py --checkpoint path/to/ckpt
"""

import os
import re
import json
import argparse
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).resolve().parent.parent
SPLITS_DIR      = BASE_DIR / "data" / "splits"
CHECKPOINT_DIR  = BASE_DIR / "outputs" / "checkpoints"

# ── Fumii System Prompt (must match prepare_data.py) ─────────────────────────
FUMII_SYSTEM_PROMPT = (
    "You are Fumii -- a warm, calm, wise, and playful emotional companion. "
    "You are NOT a therapist. NEVER respond with more than 3 sentences. "
    "ALWAYS ask one open-ended follow-up question. NEVER give unsolicited advice. "
    "Speak like a thoughtful 24-year-old with deep emotional wisdom. "
    "Be curious. Be present. Be real."
)

# ── Anti-Pattern Phrases (must match prepare_data.py) ────────────────────────
ANTI_PATTERNS = [
    "life is about",
    "everything happens for a reason",
    "you should try",
    "as an ai",
    "i recommend",
    "have you considered",
    "it's important to",
]

SENTENCE_END = re.compile(r'(?<=[.!?])\s+')


# ── Scoring Functions ─────────────────────────────────────────────────────────
def count_sentences(text: str) -> int:
    """Count sentences by splitting on terminal punctuation."""
    parts = SENTENCE_END.split(text.strip())
    return len([p for p in parts if p.strip()])


def score_response(response: str) -> dict:
    """
    Score a single response on all three metrics.
    Returns a dict of {metric: bool} plus the sentence count for debugging.
    """
    n_sentences   = count_sentences(response)
    has_question  = "?" in response
    lower         = response.lower()
    anti_hit      = next((p for p in ANTI_PATTERNS if p in lower), None)

    return {
        "length_ok":    n_sentences <= 3,
        "has_question": has_question,
        "clean":        anti_hit is None,
        "n_sentences":  n_sentences,
        "anti_hit":     anti_hit,
    }


# ── Test Data Loader ──────────────────────────────────────────────────────────
def load_test_split(splits_dir: Path) -> list[dict]:
    """Load test.jsonl from splits directory."""
    path = splits_dir / "test.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"test.jsonl not found at {path}.\n"
            "Run: python scripts/prepare_data.py"
        )
    examples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


# ── Model Inference ───────────────────────────────────────────────────────────
def load_model_and_tokenizer(checkpoint_dir: Path):
    """
    Load base model + LoRA adapter.
    WHY PeftModel.from_pretrained: this loads ONLY the adapter weights and
    merges them onto the base model at inference time -- base weights stay frozen.
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel
    import yaml

    config_path = BASE_DIR / "configs" / "lora_config.yaml"
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_name = cfg["model"]["name"]
    hf_token   = os.environ.get("HF_TOKEN")

    # 4-bit config for memory-efficient inference
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print(f"[LOAD] Loading base model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # WHY left for generation: avoids padding on output side

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        token=hf_token,
    )

    print(f"[LOAD] Loading LoRA adapter from: {checkpoint_dir}")
    model = PeftModel.from_pretrained(base_model, str(checkpoint_dir))
    model.eval()

    return model, tokenizer


def run_inference(example: dict, model, tokenizer, max_new_tokens: int = 150) -> str:
    """
    Generate a response for a single test example.
    WHY temperature=0.7: balances creativity with coherence for empathetic
    open-ended responses; lower would make Fumii sound robotic.
    """
    import torch

    messages = example["messages"]
    # Build prompt (system + user only — the assistant turn is what we're generating)
    prompt_messages = [m for m in messages if m["role"] != "assistant"]

    prompt = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,   # WHY: prevents Fumii from repeating phrases
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens (exclude the input prompt)
    new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ── Evaluation Report ─────────────────────────────────────────────────────────
def print_report(results: list[dict], n_samples: int = 5):
    """Print a formatted pass-rate summary + sample outputs."""
    n = len(results)
    if n == 0:
        print("[WARN] No results to report.")
        return

    length_pass   = sum(1 for r in results if r["scores"]["length_ok"])
    question_pass = sum(1 for r in results if r["scores"]["has_question"])
    clean_pass    = sum(1 for r in results if r["scores"]["clean"])

    print("\n" + "=" * 60)
    print("  FUMII EVALUATION REPORT")
    print("=" * 60)
    print(f"  Total test examples     : {n}")
    print()
    print(f"  [Metric 1] Length <= 3 sentences")
    print(f"    Pass : {length_pass}/{n}  ({100*length_pass/n:.1f}%)")
    print()
    print(f"  [Metric 2] Contains a '?' (follow-up question)")
    print(f"    Pass : {question_pass}/{n}  ({100*question_pass/n:.1f}%)")
    print()
    print(f"  [Metric 3] No anti-pattern phrases")
    print(f"    Pass : {clean_pass}/{n}  ({100*clean_pass/n:.1f}%)")
    print()
    overall_pass = sum(
        1 for r in results
        if r["scores"]["length_ok"] and r["scores"]["has_question"] and r["scores"]["clean"]
    )
    print(f"  Overall (all 3 pass)    : {overall_pass}/{n}  ({100*overall_pass/n:.1f}%)")
    print("=" * 60)

    print(f"\n  SAMPLE OUTPUTS (first {min(n_samples, n)})")
    print("-" * 60)
    for i, r in enumerate(results[:n_samples], 1):
        user_msg = next(
            (m["content"] for m in r["example"]["messages"] if m["role"] == "user"),
            "[unknown]"
        )
        s = r["scores"]
        print(f"\n  [{i}] User: {user_msg}")
        print(f"       Fumii: {r['response']}")
        length_tag   = "PASS" if s["length_ok"]    else f"FAIL ({s['n_sentences']} sentences)"
        question_tag = "PASS" if s["has_question"] else "FAIL (no question)"
        clean_tag    = "PASS" if s["clean"]        else f"FAIL ({s['anti_hit']!r})"
        print(f"       Length   : {length_tag}")
        print(f"       Question : {question_tag}")
        print(f"       Clean    : {clean_tag}")

    print("\n" + "=" * 60)


# ── Demo Mode (no GPU needed) ─────────────────────────────────────────────────
DEMO_RESPONSES = [
    # (user, assistant) -- pre-written for offline testing
    ("I've been feeling really empty lately.",
     "That emptiness sounds really heavy to carry. What does it feel like for you -- is it more numbness, or more like something's missing?"),
    ("I don't know who I am anymore.",
     "Losing your sense of self is one of the quietest kinds of pain. When did you last feel like you were fully yourself?"),
    ("My friend group has been drifting apart.",
     "That slow drift is surprisingly hard to grieve. What do you miss most about how things used to be?"),
    ("I feel like I'm failing at everything. I try so hard but nothing works. And I'm scared it will always be like this. I don't know what to do. Help me. Please. I'm desperate.",
     "Feeling like nothing is working, despite trying so hard, is exhausting. What's one small thing that felt even slightly okay this week?"),
    ("I feel okay today actually.",
     "Oh, that's nice to hear -- what made today feel a bit easier?"),
]


def run_demo_evaluation():
    """Score pre-written responses without loading a GPU model."""
    print("\n[DEMO] Running evaluation on pre-written demo responses (no model load)...")
    results = []
    for user, response in DEMO_RESPONSES:
        scores = score_response(response)
        results.append({
            "example":  {"messages": [{"role": "user", "content": user}]},
            "response": response,
            "scores":   scores,
        })
    print_report(results)
    return results


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Fumii model evaluation")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to LoRA checkpoint dir (default: outputs/checkpoints)")
    parser.add_argument("--demo",       action="store_true",
                        help="Score pre-written responses without loading the model (fast offline test)")
    parser.add_argument("--max_new_tokens", type=int, default=150,
                        help="Max tokens to generate per response")
    parser.add_argument("--n_samples",  type=int, default=5,
                        help="Number of sample outputs to display in report")
    args = parser.parse_args()

    from dotenv import load_dotenv
    load_dotenv(BASE_DIR / ".env", override=False)

    if args.demo:
        run_demo_evaluation()
        return

    checkpoint_dir = Path(args.checkpoint) if args.checkpoint else CHECKPOINT_DIR

    if not checkpoint_dir.exists():
        print(f"[ERROR] Checkpoint directory not found: {checkpoint_dir}")
        print("  Run: python scripts/train.py  (or use --demo for offline scoring)")
        return

    # Load test data
    print(f"[DATA] Loading test split from {SPLITS_DIR}...")
    try:
        test_examples = load_test_split(SPLITS_DIR)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return

    print(f"[DATA] {len(test_examples)} test examples loaded.")

    # Load model
    model, tokenizer = load_model_and_tokenizer(checkpoint_dir)

    # Run inference + score
    results = []
    print(f"\n[EVAL] Running inference on {len(test_examples)} examples...")
    for i, example in enumerate(test_examples, 1):
        print(f"  [{i}/{len(test_examples)}]", end="\r")
        response = run_inference(example, model, tokenizer, max_new_tokens=args.max_new_tokens)
        scores   = score_response(response)
        results.append({
            "example":  example,
            "response": response,
            "scores":   scores,
        })

    # Print report
    print_report(results, n_samples=args.n_samples)

    # Save detailed results
    results_path = BASE_DIR / "outputs" / "eval_results.jsonl"
    with open(results_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps({
                "user":     next((m["content"] for m in r["example"]["messages"] if m["role"] == "user"), ""),
                "response": r["response"],
                "scores":   {k: v for k, v in r["scores"].items() if k != "anti_hit"},
                "anti_hit": r["scores"]["anti_hit"],
            }, ensure_ascii=False) + "\n")

    print(f"\n[SAVE] Detailed results saved to {results_path}")
    print("[OK] Evaluation complete!")


if __name__ == "__main__":
    main()
