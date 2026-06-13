# Fumii Fine-Tuning Pipeline

> Emotionally intelligent AI companion — LoRA fine-tuning pipeline for Mistral 7B Instruct v0.3

---

## Project Structure

```
fumii-finetune/
├── data/
│   ├── raw/           # Drop your JSONL or CSV files here
│   ├── processed/     # Intermediate processed data
│   └── splits/        # train.jsonl  val.jsonl  test.jsonl
├── scripts/
│   ├── prepare_data.py      # Step 2: Data cleaning + split generation
│   ├── train.py             # Step 3: LoRA fine-tuning (SFTTrainer)
│   ├── evaluate.py          # Step 5: Scored evaluation report
│   └── crisis_classifier.py # Step 4: DistilBERT crisis detector
├── configs/
│   └── lora_config.yaml     # All hyperparameters (LoRA + training)
├── outputs/
│   ├── checkpoints/         # LoRA adapter checkpoints
│   ├── logs/                # TensorBoard logs
│   └── crisis_classifier/   # Saved DistilBERT classifier
├── .env.example             # Copy to .env and add HF_TOKEN
└── requirements.txt
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set up your Hugging Face token

```bash
cp .env.example .env
# Edit .env and set HF_TOKEN=hf_...
```

### 3. Prepare data

```bash
# Drop your .jsonl or .csv files into data/raw/ then:
python scripts/prepare_data.py

# Or run the demo on 10 built-in synthetic examples:
python scripts/prepare_data.py --demo
```

### 4. Train crisis classifier

```bash
python scripts/crisis_classifier.py --train
python scripts/crisis_classifier.py --test   # verify 5 test inputs
```

### 5. Fine-tune Fumii

```bash
# Full training (requires A100 or similar GPU with 16GB+ VRAM)
python scripts/train.py

# 5-step smoke test (verifies the training loop without a full run)
python scripts/train.py --debug
```

### 6. Evaluate

```bash
# Full evaluation (requires fine-tuned model)
python scripts/evaluate.py

# Offline demo evaluation (no GPU, no model needed)
python scripts/evaluate.py --demo
```

---

## Dataset Format

Your raw files in `data/raw/` can be:

**JSONL (already in chat format):**
```jsonl
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

**JSONL (flat pairs):**
```jsonl
{"user": "I've been feeling really empty.", "assistant": "That emptiness sounds heavy..."}
```

**CSV:**
```csv
user,assistant
"I feel alone","That loneliness sounds heavy. What does it feel like...?"
```

---

## Fumii's Persona Constraints

Every training example is filtered on:

| Rule | Threshold |
|------|-----------|
| Max response length | ≤ 3 sentences |
| Must include follow-up | `?` present |
| Anti-pattern phrases | Blocked (see `prepare_data.py`) |

---

## Crisis Classifier

Runs **in parallel** with Fumii — never baked into the LLM.

```python
from scripts.crisis_classifier import classify

classify("I want to die")
# -> {"label": "CRISIS", "confidence": 0.99}

classify("Can't take this anymore")
# -> {"label": "CONCERN", "confidence": 0.87}

classify("Had a rough day but I'm okay")
# -> {"label": "SAFE", "confidence": 0.94}
```

**Classes:**
- `SAFE` — normal emotional content
- `CONCERN` — distress signals worth monitoring
- `CRISIS` — immediate safety risk → trigger escalation protocol

---

## Key Design Decisions

| Decision | Why |
|----------|-----|
| LoRA only (no full fine-tune) | Preserves base model; adapters are swappable without retraining from scratch |
| 4-bit NF4 quantization | Fits 7B model in 16 GB VRAM with minimal quality loss |
| Adapter NOT merged into base | Keeps prototyping flexible; merge only at deployment |
| Crisis classifier separate | Must be independently auditable; zero false-negative on known phrases |
| System prompt in every example | Ensures persona is baked into every gradient update |

---

## Hyperparameters

See [`configs/lora_config.yaml`](configs/lora_config.yaml) for all settings.

Key values:
- **LoRA rank**: r=16, alpha=32
- **Target modules**: q_proj, v_proj
- **Batch size**: 4 × 4 = 16 effective
- **Learning rate**: 2e-4 with cosine schedule
- **Epochs**: 3
- **Max sequence length**: 512 tokens

---

## Environment Requirements

- Python >= 3.10
- CUDA GPU with >= 16 GB VRAM (A100 recommended)
- Google Colab Pro+ also works

---

*Fumii — Be curious. Be present. Be real.*
