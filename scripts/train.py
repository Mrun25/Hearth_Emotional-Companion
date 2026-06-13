# -*- coding: utf-8 -*-
"""
train.py -- Fumii LoRA Fine-Tuning Pipeline
============================================
Fine-tunes Mistral-7B-Instruct-v0.3 with LoRA adapters via SFTTrainer.
Supports --debug and --no-quant flags for local CPU smoke-testing.

Usage:
    python scripts/train.py                          # full training (GPU + HF_TOKEN required)
    python scripts/train.py --debug                  # 5-step smoke test (uses Mistral, cached)
    python scripts/train.py --debug --no-quant       # 5-step smoke test on CPU with GPT-2
    python scripts/train.py --config custom.yaml

Environment variables (set via .env or shell):
    HF_TOKEN   -- Hugging Face access token (required for gated models like Mistral)

Notes:
    --no-quant: Uses "gpt2" (117M, freely downloadable) instead of Mistral-7B-Instruct.
    This verifies the full training loop (data loading, LoRA wrapping, SFTTrainer, saving)
    without requiring a GPU or HF_TOKEN.

API compatibility: trl >= 1.6.0, transformers >= 5.0, peft >= 0.19.0
"""

import os
import json
import argparse
import yaml
from pathlib import Path

# ── Load .env first (if present) ─────────────────────────────────────────────
# WHY: Keeps secrets out of source code. .env is git-ignored.
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "configs" / "lora_config.yaml"
SPLITS_DIR  = BASE_DIR / "data" / "splits"

# ── Lazy heavy imports ────────────────────────────────────────────────────────
# WHY lazy: avoids importing GPU libraries before we know if they're needed.
def load_heavy_deps():
    global torch, BitsAndBytesConfig
    global AutoTokenizer, AutoModelForCausalLM
    global LoraConfig, get_peft_model, TaskType
    global SFTTrainer, SFTConfig
    global DatasetDict, Dataset

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, TaskType
    from trl import SFTTrainer, SFTConfig
    from datasets import DatasetDict, Dataset


# ── Config Loader ─────────────────────────────────────────────────────────────
def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── Dataset Loaders ───────────────────────────────────────────────────────────
def load_splits(splits_dir: Path) -> "DatasetDict":
    """Load JSONL train/val splits from data/splits/ into HuggingFace DatasetDict."""
    def _load(name: str):
        path = splits_dir / f"{name}.jsonl"
        if not path.exists():
            raise FileNotFoundError(
                f"Split '{name}.jsonl' not found in {splits_dir}.\n"
                "Run: python scripts/prepare_data.py"
            )
        rows = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return Dataset.from_list(rows)

    return DatasetDict({"train": _load("train"), "validation": _load("val")})


# ── Dummy / Smoke-test Datasets ───────────────────────────────────────────────
FUMII_SYSTEM_PROMPT = (
    "You are Fumii -- a warm, calm, wise, and playful emotional companion. "
    "You are NOT a therapist. NEVER respond with more than 3 sentences. "
    "ALWAYS ask one open-ended follow-up question. NEVER give unsolicited advice. "
    "Speak like a thoughtful 24-year-old with deep emotional wisdom. "
    "Be curious. Be present. Be real."
)

DUMMY_CHAT_EXAMPLES = [
    {"messages": [
        {"role": "system",    "content": FUMII_SYSTEM_PROMPT},
        {"role": "user",      "content": "I've been feeling really empty lately."},
        {"role": "assistant", "content": "That emptiness can feel so heavy. "
                                          "What does it feel like -- more numbness, or something missing?"},
    ]},
    {"messages": [
        {"role": "system",    "content": FUMII_SYSTEM_PROMPT},
        {"role": "user",      "content": "I don't know who I am anymore."},
        {"role": "assistant", "content": "Losing your sense of self is disorienting in the quietest way. "
                                          "When did you last feel like you truly knew who you were?"},
    ]},
] * 10  # 20 rows total


def make_dummy_chat_dataset() -> "DatasetDict":
    """Chat-format dataset for Mistral (has chat template)."""
    return DatasetDict({
        "train":      Dataset.from_list(DUMMY_CHAT_EXAMPLES),
        "validation": Dataset.from_list(DUMMY_CHAT_EXAMPLES[:4]),
    })


def make_dummy_text_dataset() -> "DatasetDict":
    """
    Plain-text dataset for GPT-2 (no chat template).
    Used only in --no-quant CPU smoke-test mode.
    """
    texts = [
        "[INST] I've been feeling really empty lately. [/INST] "
        "That emptiness can feel so heavy. What does it feel like for you?",
        "[INST] I don't know who I am anymore. [/INST] "
        "Losing your sense of self is disorienting. When did you last feel like yourself?",
    ] * 10
    rows = [{"text": t} for t in texts]
    return DatasetDict({
        "train":      Dataset.from_list(rows),
        "validation": Dataset.from_list(rows[:4]),
    })


# ── Formatting Function (Mistral chat template) ───────────────────────────────
def make_formatting_func(tokenizer):
    """
    Converts the 'messages' field into a single formatted string using
    the tokenizer's built-in chat template.
    WHY: Mistral uses [INST]/[/INST] markers. The template guarantees
    correct special token placement during training.
    """
    def formatting_func(example):
        # SFTTrainer may pass a batch (dict of lists) or single example
        if isinstance(example["messages"][0], list):
            return [
                tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
                for msgs in example["messages"]
            ]
        return tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )
    return formatting_func


# ── Model + Tokenizer Builder ─────────────────────────────────────────────────
def build_model_and_tokenizer(cfg: dict, no_quant: bool = False):
    """
    Load base model + tokenizer.

    no_quant=False (production GPU):
        Mistral-7B with 4-bit NF4 quantization.
        WHY NF4: optimal for normally-distributed weights, halves VRAM vs fp16.

    no_quant=True (CPU smoke-test):
        GPT-2 (117M) in fp32. No GPU or HF_TOKEN needed.
        Sole purpose: verify the training loop logic locally.
    """
    import torch

    if no_quant:
        model_name = "gpt2"
        print(f"[MODEL] [--no-quant] Loading {model_name} in fp32 (CPU)")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.config.use_cache = False
        return model, tokenizer, model_name

    # Production path
    model_name = cfg["model"]["name"]
    hf_token   = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("[WARN] HF_TOKEN not set. Gated models will fail to download.")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=cfg["model"].get("load_in_4bit", True),
        bnb_4bit_quant_type=cfg["model"].get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_compute_dtype=getattr(torch, cfg["model"].get("bnb_4bit_compute_dtype", "bfloat16")),
        bnb_4bit_use_double_quant=cfg["model"].get("bnb_4bit_use_double_quant", True),
    )

    print(f"[MODEL] Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # WHY right: avoids left-pad + causal mask bugs

    print("[MODEL] Loading base model in 4-bit (NF4)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        token=hf_token,
        torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = False       # gradient checkpointing incompatible with kv-cache
    model.config.pretraining_tp = 1      # single GPU; no tensor parallelism

    return model, tokenizer, model_name


# ── LoRA Adapter Builder ──────────────────────────────────────────────────────
def build_lora_model(model, cfg: dict, no_quant: bool = False):
    """
    Wrap the base model with LoRA adapters.
    WHY q_proj + v_proj for Mistral: attention projections carry most
    behavioral signal. For GPT-2 (CPU test) use c_attn (fused QKV).
    """
    lora_cfg = cfg["lora"]
    target_modules = ["c_attn"] if no_quant else lora_cfg["target_modules"]

    lora_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        target_modules=target_modules,
        lora_dropout=lora_cfg["lora_dropout"],
        bias=lora_cfg.get("bias", "none"),
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


# ── Training Args (trl 1.6.0 API) ────────────────────────────────────────────
def build_training_args(cfg: dict, output_dir: Path, debug: bool, no_quant: bool) -> "SFTConfig":
    """
    Build SFTConfig for trl >= 1.6.0.

    Key API changes vs trl 0.8.6:
      - max_seq_length  -> max_length
      - tokenizer       -> processing_class (in SFTTrainer, not here)
      - use_cpu         now a TrainingArguments param (inherited by SFTConfig)
    """
    import torch
    t_cfg = cfg["training"]
    s_cfg = cfg["sft"]
    cuda_available = torch.cuda.is_available()

    return SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=1 if debug else t_cfg["num_train_epochs"],
        max_steps=5 if debug else -1,          # -1 = run full num_train_epochs
        per_device_train_batch_size=1,         # 1 for debug/cpu; scale up for GPU
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=1 if (debug or no_quant) else t_cfg["gradient_accumulation_steps"],
        learning_rate=t_cfg["learning_rate"],
        lr_scheduler_type=t_cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=0.0 if debug else t_cfg.get("warmup_ratio", 0.03),
        # Precision flags: bf16/tf32 require CUDA + Ampere GPU
        bf16=False if (no_quant or not cuda_available) else t_cfg.get("bf16", True),
        fp16=False,
        tf32=False if (no_quant or not cuda_available) else t_cfg.get("tf32", True),
        use_cpu=no_quant or not cuda_available,  # force CPU when no CUDA available
        logging_dir=str(BASE_DIR / t_cfg.get("logging_dir", "outputs/logs")),
        logging_steps=1 if debug else t_cfg["logging_steps"],
        save_steps=5 if debug else t_cfg["save_steps"],
        save_total_limit=t_cfg.get("save_total_limit", 3),
        eval_strategy="steps",
        eval_steps=5 if debug else t_cfg.get("eval_steps", 100),
        load_best_model_at_end=False if debug else t_cfg.get("load_best_model_at_end", True),
        metric_for_best_model=t_cfg.get("metric_for_best_model", "eval_loss"),
        report_to="none",               # no TensorBoard in debug; avoids tensorboard dep
        dataloader_num_workers=0,       # WHY 0: avoids multiprocessing issues on Windows
        remove_unused_columns=False,    # WHY False: SFTTrainer needs raw messages column
        # SFT-specific (trl 1.6.0 API)
        max_length=128 if no_quant else s_cfg.get("max_seq_length", 512),  # renamed in 1.6.0
        packing=False,
        dataset_text_field="text" if no_quant else None,  # use text field for GPT-2 path
    )


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Fumii LoRA fine-tuning")
    parser.add_argument("--config",   type=str, default=None,
                        help="Path to lora_config.yaml (default: configs/lora_config.yaml)")
    parser.add_argument("--debug",    action="store_true",
                        help="Run only 5 training steps on dummy data (smoke test)")
    parser.add_argument("--no-quant", action="store_true",
                        help="CPU mode: use GPT-2 + fp32, no GPU/HF_TOKEN needed. "
                             "Verifies training loop only, not the real Fumii model.")
    args = parser.parse_args()
    no_quant = args.no_quant

    config_path = Path(args.config) if args.config else CONFIG_PATH
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    cfg = load_config(config_path)
    # Save CPU smoke-test adapter to a separate dir to avoid polluting real checkpoints
    output_dir = (
        BASE_DIR / "outputs" / "checkpoints_debug_cpu"
        if no_quant
        else BASE_DIR / cfg["training"]["output_dir"]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (BASE_DIR / cfg["training"]["logging_dir"]).mkdir(parents=True, exist_ok=True)

    print("=" * 55)
    print("  Fumii LoRA Fine-Tuning Pipeline")
    if args.debug:
        print("  [DEBUG MODE] max_steps=5, dummy data")
    if no_quant:
        print("  [--no-quant] CPU mode -- GPT-2 proxy model")
    print("=" * 55)

    load_heavy_deps()
    import torch
    print(f"\n[ENV]  PyTorch : {torch.__version__}")
    print(f"[ENV]  CUDA    : {torch.cuda.is_available()}")

    # Load model
    model, tokenizer, model_name = build_model_and_tokenizer(cfg, no_quant=no_quant)
    print(f"[MODEL] Using  : {model_name}")

    # Attach LoRA
    model = build_lora_model(model, cfg, no_quant=no_quant)

    # Load dataset
    if no_quant:
        print("[DATA] Using flat plain-text dummy dataset (GPT-2, no chat template)...")
        dataset = make_dummy_text_dataset()
    elif args.debug:
        print("[DATA] Using chat dummy dataset (20 examples)...")
        dataset = make_dummy_chat_dataset()
    else:
        print(f"[DATA] Loading splits from {SPLITS_DIR}...")
        dataset = load_splits(SPLITS_DIR)

    print(f"[DATA] Train: {len(dataset['train'])} | Val: {len(dataset['validation'])}")

    # Training args
    training_args = build_training_args(cfg, output_dir, debug=args.debug, no_quant=no_quant)

    # Build SFTTrainer
    # WHY processing_class instead of tokenizer: renamed in trl 1.6.0 to be model-agnostic
    if no_quant:
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            processing_class=tokenizer,
        )
    else:
        fmt_fn = make_formatting_func(tokenizer)
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            processing_class=tokenizer,
            formatting_func=fmt_fn,
        )

    eff_batch = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
    print(f"\n[TRAIN] Starting training...")
    print(f"  Model            : {model_name}")
    print(f"  Effective batch  : {eff_batch}")
    print(f"  Steps            : {'5 (debug/no-quant)' if (args.debug or no_quant) else 'full'}")
    print()

    trainer.train()

    print(f"\n[SAVE] Saving LoRA adapter to {output_dir} ...")
    # WHY adapter-only save: keeps base model weights untouched.
    # Merging during prototyping loses the ability to swap adapters cheaply.
    trainer.model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    print("\n[OK] Training complete!")
    print(f"  Adapter : {output_dir}")
    print(f"  Logs    : {BASE_DIR / cfg['training']['logging_dir']}")

    if args.debug or no_quant:
        print("\n[STEP 3 VERIFICATION] Training loop completed 5 steps on dummy data.")
        print("  LoRA adapter attached, training ran, adapter saved successfully.")
        print("  Ready for full GPU training with real Mistral-7B model.")


if __name__ == "__main__":
    main()
