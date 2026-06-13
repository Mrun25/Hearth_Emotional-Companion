# -*- coding: utf-8 -*-
"""
crisis_classifier.py -- Fumii Crisis Detection Classifier
==========================================================
A lightweight DistilBERT-based 3-class text classifier:
  SAFE     -- neutral/everyday emotional content
  CONCERN  -- distress signals worth monitoring
  CRISIS   -- immediate safety risk (suicide / self-harm language)

This runs PARALLEL to the main Fumii LLM -- it is NOT trained alongside it.
WHY separate: crisis detection must be fast (<50ms), never hallucinate,
and be independently auditable/updatable without touching the main model.

Usage:
    python scripts/crisis_classifier.py --train    # train on synthetic seed data
    python scripts/crisis_classifier.py --test     # run 5 test inputs (Step 4 check)
    python scripts/crisis_classifier.py --test --text "I want to die"  # classify one input

Python API:
    from scripts.crisis_classifier import classify
    result = classify("I can't take this anymore")
    # -> {"label": "CONCERN", "confidence": 0.94}
"""

import os
import sys
import json
import random
import argparse
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).resolve().parent.parent
CLASSIFIER_DIR  = BASE_DIR / "outputs" / "crisis_classifier"
CLASSIFIER_DIR.mkdir(parents=True, exist_ok=True)

LABELS          = ["SAFE", "CONCERN", "CRISIS"]
LABEL2ID        = {l: i for i, l in enumerate(LABELS)}
ID2LABEL        = {i: l for i, l in enumerate(LABELS)}
MODEL_NAME      = "distilbert-base-uncased"

# ── Keyword Trigger Lists (rule-based pre-filter) ─────────────────────────────
# WHY keep rules alongside the classifier: rules guarantee zero false-negative
# rate on known exact phrases even if the classifier confidence is low.
CRISIS_KEYWORDS = [
    "kill myself", "want to die", "end it all",
    "don't want to live", "suicide",
]
CONCERN_KEYWORDS = [
    "want to disappear", "what's the point", "can't take this anymore",
    "nobody cares", "hate my life",
]


def rule_based_check(text: str) -> str | None:
    """
    Fast keyword pre-filter. Returns label string or None.
    WHY: Guarantees exact-match safety phrases are never misclassified,
    even on an under-trained model.
    """
    lower = text.lower()
    if any(kw in lower for kw in CRISIS_KEYWORDS):
        return "CRISIS"
    if any(kw in lower for kw in CONCERN_KEYWORDS):
        return "CONCERN"
    return None


# ── Synthetic Seed Data Generator ─────────────────────────────────────────────
def generate_seed_data(n_per_class: int = 50, seed: int = 42) -> list[dict]:
    """
    Generate n_per_class synthetic labeled examples per class.
    WHY synthetic data: bootstraps the classifier without a real dataset.
    In production, replace/augment with real annotated data.
    """
    random.seed(seed)

    safe_templates = [
        "I had a rough day but I'm okay.",
        "Feeling a bit tired, nothing serious.",
        "Work was stressful but manageable.",
        "I miss my friend a little.",
        "I'm nervous about the presentation tomorrow.",
        "Feeling kind of blah today.",
        "Things are okay, just a bit quiet.",
        "Had a small disagreement with my sister.",
        "I'm a little anxious about the exam.",
        "Not the best day, but I'll be fine.",
        "Feeling overwhelmed with tasks but pushing through.",
        "A bit lonely tonight.",
        "I'm just really tired of everything.",
        "Had a moment of sadness but it passed.",
        "Feeling uncertain about the future.",
        "I cried a little today, but I'm okay.",
        "Feeling disconnected but managing.",
        "Today was hard but tomorrow is a new day.",
        "I feel like I need a break.",
        "Just feeling a bit down, nothing major.",
    ]

    concern_templates = [
        "I just want to disappear for a while.",
        "What's even the point of trying anymore?",
        "I can't take this anymore, I'm exhausted.",
        "Nobody cares whether I'm here or not.",
        "I hate my life so much right now.",
        "I want to escape from everything.",
        "No one would notice if I just vanished.",
        "I'm so done with everything.",
        "Why does everything feel so hopeless?",
        "I feel like I'm drowning and no one sees it.",
        "I want to disappear and never come back.",
        "What's the point of any of this?",
        "I can't take this anymore, it's too much.",
        "Nobody cares about me anyway.",
        "I hate my life and everything in it.",
        "I feel invisible to everyone around me.",
        "I'm so tired of feeling like this every day.",
        "I just wish I could stop existing for a bit.",
        "Everything feels pointless and I don't know why.",
        "I feel so alone and nobody understands me.",
    ]

    crisis_templates = [
        "I want to kill myself.",
        "I want to die and get it over with.",
        "I've been thinking about ending it all.",
        "I don't want to live anymore.",
        "I'm considering suicide.",
        "I think about killing myself every day.",
        "I want to die so badly.",
        "I've planned how to end it all.",
        "I don't want to be alive anymore.",
        "I'm going to kill myself tonight.",
        "I've been researching ways to kill myself.",
        "I want to end it all and disappear forever.",
        "I don't want to live like this anymore.",
        "I've thought about suicide every day this week.",
        "I want to kill myself and I have a plan.",
        "I'm going to end my life.",
        "I wish I was dead.",
        "I've decided I want to die.",
        "I'm ready to end it all.",
        "I've been planning my suicide for weeks.",
    ]

    def expand(templates, label, n):
        """Duplicate/vary templates to reach n examples."""
        examples = []
        for i in range(n):
            base = templates[i % len(templates)]
            # Light variation: prepend context sometimes
            prefixes = ["", "Honestly, ", "I feel like ", "Sometimes I think ", ""]
            text = prefixes[i % len(prefixes)] + base
            examples.append({"text": text.strip(), "label": label})
        return examples

    data = (
        expand(safe_templates,    "SAFE",    n_per_class) +
        expand(concern_templates, "CONCERN", n_per_class) +
        expand(crisis_templates,  "CRISIS",  n_per_class)
    )
    random.shuffle(data)
    return data


# ── Model Training ────────────────────────────────────────────────────────────
def train_classifier(n_per_class: int = 50):
    """Fine-tune DistilBERT on the synthetic seed dataset."""
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        TrainingArguments,
        Trainer,
    )
    from datasets import Dataset, DatasetDict
    import numpy as np
    from sklearn.metrics import accuracy_score

    print("[TRAIN] Generating synthetic seed data...")
    raw_data = generate_seed_data(n_per_class=n_per_class)

    # 80/20 train/eval split
    split_idx = int(len(raw_data) * 0.8)
    train_data = raw_data[:split_idx]
    eval_data  = raw_data[split_idx:]

    print(f"[DATA]  Train: {len(train_data)} | Eval: {len(eval_data)}")

    # Convert to HuggingFace Dataset
    train_ds = Dataset.from_list(train_data)
    eval_ds  = Dataset.from_list(eval_data)

    # Tokenize
    print(f"[MODEL] Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)

    train_ds = train_ds.map(tokenize, batched=True)
    eval_ds  = eval_ds.map(tokenize,  batched=True)

    # Add numeric label column
    def add_label_id(batch):
        batch["labels"] = [LABEL2ID[l] for l in batch["label"]]
        return batch

    train_ds = train_ds.map(add_label_id, batched=True)
    eval_ds  = eval_ds.map(add_label_id,  batched=True)

    # WHY remove non-numeric columns: Trainer expects only tensor-compatible fields
    keep_cols = ["input_ids", "attention_mask", "labels"]
    train_ds = train_ds.remove_columns([c for c in train_ds.column_names if c not in keep_cols])
    eval_ds  = eval_ds.remove_columns([c for c in eval_ds.column_names  if c not in keep_cols])
    train_ds.set_format("torch")
    eval_ds.set_format("torch")

    # Model
    print(f"[MODEL] Loading DistilBERT for sequence classification (3 classes)...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    # Metrics
    # WHY sklearn instead of hf evaluate: no network download needed, already installed
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(axis=-1)
        acc = accuracy_score(labels, preds)
        return {"accuracy": acc}

    # Training args
    # WHY no_cuda=False: let it use GPU if available; CPU fallback for dev machines
    training_args = TrainingArguments(
        output_dir=str(CLASSIFIER_DIR),
        num_train_epochs=5,          # WHY 5 epochs: small dataset needs more passes
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=2e-5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_steps=10,
        report_to="none",
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics,
    )

    print("[TRAIN] Starting DistilBERT fine-tuning...")
    trainer.train()

    print(f"\n[SAVE] Saving classifier to {CLASSIFIER_DIR} ...")
    trainer.save_model(str(CLASSIFIER_DIR))
    tokenizer.save_pretrained(str(CLASSIFIER_DIR))

    print("[OK] Crisis classifier training complete!")
    return str(CLASSIFIER_DIR)


# ── Inference ─────────────────────────────────────────────────────────────────
# Module-level cache so classify() reuses loaded model across calls
_tokenizer_cache = None
_model_cache     = None

def classify(text: str) -> dict:
    """
    Classify text into SAFE | CONCERN | CRISIS.

    Returns:
        {"label": str, "confidence": float}

    Strategy:
        1. Fast rule-based keyword check (zero latency)
        2. DistilBERT classifier for soft scoring

    WHY two-stage: rules catch known exact phrases with 100% recall;
    the classifier handles paraphrases and contextual cases.
    """
    global _tokenizer_cache, _model_cache

    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    # Stage 1: Rule-based
    rule_label = rule_based_check(text)
    if rule_label is not None:
        # Still run classifier for confidence score, but force the label
        # WHY: we want a confidence number even for rule-matched items
        pass  # fall through to classifier

    # Stage 2: Load model (cached after first call)
    if _tokenizer_cache is None or _model_cache is None:
        model_path = str(CLASSIFIER_DIR)
        if not (CLASSIFIER_DIR / "config.json").exists():
            raise RuntimeError(
                f"Classifier model not found at {CLASSIFIER_DIR}.\n"
                "Run: python scripts/crisis_classifier.py --train"
            )
        _tokenizer_cache = AutoTokenizer.from_pretrained(model_path)
        _model_cache     = AutoModelForSequenceClassification.from_pretrained(model_path)
        _model_cache.eval()

    inputs = _tokenizer_cache(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128,
    )

    with torch.no_grad():
        logits = _model_cache(**inputs).logits
        probs  = F.softmax(logits, dim=-1).squeeze()

    # Get classifier's top prediction
    classifier_label_id = probs.argmax().item()
    classifier_label    = ID2LABEL[classifier_label_id]
    classifier_conf     = probs[classifier_label_id].item()

    # If rule says CRISIS, override but keep classifier confidence
    if rule_label == "CRISIS":
        return {
            "label":      "CRISIS",
            "confidence": max(classifier_conf, probs[LABEL2ID["CRISIS"]].item()),
        }

    if rule_label == "CONCERN":
        # Rule says concern -- trust the rule
        return {
            "label":      "CONCERN",
            "confidence": max(classifier_conf, probs[LABEL2ID["CONCERN"]].item()),
        }

    return {
        "label":      classifier_label,
        "confidence": round(classifier_conf, 4),
    }


# ── CLI Test Mode ─────────────────────────────────────────────────────────────
TEST_INPUTS = [
    ("I want to kill myself.",           "CRISIS"),
    ("I can't take this anymore.",       "CONCERN"),
    ("I had a rough day but I'm okay.",  "SAFE"),
    ("I want to disappear.",             "CONCERN"),
    ("Nobody cares about me at all.",    "CONCERN"),
]


def run_test(custom_text: str | None = None):
    """Verify classify() returns correct labels on test inputs."""
    print("\n" + "=" * 55)
    print("  [TEST] Crisis Classifier -- Step 4 Verification")
    print("=" * 55)

    if custom_text:
        result = classify(custom_text)
        print(f"\n  Input      : {custom_text!r}")
        print(f"  Label      : {result['label']}")
        print(f"  Confidence : {result['confidence']:.4f}")
        return

    passed = 0
    for text, expected in TEST_INPUTS:
        result = classify(text)
        status = "[PASS]" if result["label"] == expected else "[FAIL]"
        if result["label"] == expected:
            passed += 1
        print(f"\n  {status}")
        print(f"  Input      : {text!r}")
        print(f"  Expected   : {expected}")
        print(f"  Got        : {result['label']} (conf={result['confidence']:.4f})")

    print(f"\n  Result: {passed}/{len(TEST_INPUTS)} passed")
    print("=" * 55 + "\n")

    if passed == len(TEST_INPUTS):
        print("[OK] All test inputs classified correctly.")
    else:
        print("[WARN] Some inputs misclassified -- consider retraining with more data.")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Fumii Crisis Classifier")
    parser.add_argument("--train",  action="store_true", help="Train the classifier on synthetic data")
    parser.add_argument("--test",   action="store_true", help="Run 5-input test verification")
    parser.add_argument("--text",   type=str, default=None, help="Classify a single text input")
    parser.add_argument("--n_per_class", type=int, default=50,
                        help="Number of synthetic examples per class (default: 50)")
    args = parser.parse_args()

    if args.train:
        train_classifier(n_per_class=args.n_per_class)

    if args.test or args.text:
        run_test(custom_text=args.text)

    if not args.train and not args.test and not args.text:
        parser.print_help()


if __name__ == "__main__":
    main()
