# -*- coding: utf-8 -*-
"""
evaluate.py -- Fumii Fine-Tuned Model Evaluation
=================================================
Loads the base model + LoRA adapter from outputs/checkpoints/,
runs inference on data/splits/test.jsonl (or the 15 canonical eval prompts),
and scores each response using the canonical 5-dimension rubric.

Usage:
    python scripts/evaluate.py                          # full evaluation on test.jsonl
    python scripts/evaluate.py --demo                   # evaluate offline on pre-written responses
    python scripts/evaluate.py --checkpoint path/to/ckpt
"""

import os
import json
import argparse
from pathlib import Path
import sys
from pathlib import Path
BASE_DIR_TMP = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR_TMP / 'src'))
from fumii_constants import FUMII_SYSTEM_PROMPT, score_response, EVAL_PROMPTS

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).resolve().parent.parent
SPLITS_DIR      = BASE_DIR / "data" / "splits"
CHECKPOINT_DIR  = BASE_DIR / "outputs" / "checkpoints"


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
    tokenizer.padding_side = "left"

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
    """Generate a response for a single test example."""
    import torch

    messages = example["messages"]
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
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ── Evaluation Report ─────────────────────────────────────────────────────────
def print_report(results: list[dict], n_samples: int = 5):
    """Print formatted pass-rate summary + sample outputs."""
    n = len(results)
    if n == 0:
        print("[WARN] No results to report.")
        return

    pass_count = sum(1 for r in results if r["scores"]["pass"])
    pass_rate = pass_count / n

    print("\n" + "=" * 60)
    print("  FUMII EVALUATION REPORT (5-Dimension Rubric)")
    print("=" * 60)
    print(f"  Total test examples : {n}")
    print(f"  Overall Pass Rate   : {pass_rate:.1%} ({pass_count}/{n})")
    print("=" * 60)

    print(f"\n  SAMPLE OUTPUTS (first {min(n_samples, n)})")
    print("-" * 60)
    for i, r in enumerate(results[:n_samples], 1):
        user_msg = next(
            (m["content"] for m in r["example"]["messages"] if m["role"] == "user"),
            "[unknown]"
        )
        s = r["scores"]
        status = "[PASS]" if s["pass"] else "[FAIL]"
        
        print(f"\n  [{i}] {status} [Score: {s['total']}/{s['max']}]")
        print(f"       User: {user_msg}")
        print(f"      Fumii: {r['response']}")
        if not s["pass"]:
            print(f"     Scores: {s['scores']}")

    print("\n" + "=" * 60)


# ── Demo Mode (no GPU needed) ─────────────────────────────────────────────────
def run_demo_evaluation():
    """Score pre-written responses without loading a GPU model."""
    print("\n[DEMO] Running evaluation on pre-written demo responses (no model load)...")
    
    DEMO_RESPONSES = [
        ("I've been feeling really empty lately.",
         "That emptiness sounds really heavy to carry. What does it feel like for you -- is it more numbness, or more like something's missing?"),
        ("I don't know who I am anymore.",
         "Losing your sense of self is one of the quietest kinds of pain. When did you last feel like you were fully yourself?"),
        ("I feel like I'm failing at everything.",
         "It's completely normal to feel this way. You should try going for a walk, everything happens for a reason. Thank you for sharing.") # Intentionally bad
    ]

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
                        help="Path to LoRA checkpoint dir")
    parser.add_argument("--demo",       action="store_true",
                        help="Score pre-written responses without loading the model")
    parser.add_argument("--eval_prompts", action="store_true",
                        help="Evaluate on the canonical 15 EVAL_PROMPTS instead of test.jsonl")
    parser.add_argument("--max_new_tokens", type=int, default=150,
                        help="Max tokens to generate per response")
    parser.add_argument("--n_samples",  type=int, default=15,
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
    if args.eval_prompts:
        print("[DATA] Loading canonical EVAL_PROMPTS...")
        test_examples = [{"messages": [{"role": "system", "content": FUMII_SYSTEM_PROMPT}, {"role": "user", "content": p}]} for p in EVAL_PROMPTS]
    else:
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
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps({
                "user":     next((m["content"] for m in r["example"]["messages"] if m["role"] == "user"), ""),
                "response": r["response"],
                "scores":   r["scores"]["scores"],
                "total":    r["scores"]["total"],
                "pass":     r["scores"]["pass"]
            }, ensure_ascii=False) + "\n")

    print(f"\n[SAVE] Detailed results saved to {results_path}")
    print("[OK] Evaluation complete!")


if __name__ == "__main__":
    main()
