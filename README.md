# Hearth Companion Ecosystem

> An emotionally intelligent AI companion designed to provide grounded, non-clinical support. Featuring an **Agentic Self-Correction Loop**, lightning-fast inference via Groq, and a modern glassmorphic web interface.

---

## 🌟 What is Hearth?

Hearth (formerly Fumii) is an experimental LLM companion that breaks away from the typical "helpful AI assistant" persona. It is not a therapist, and it doesn't give advice or bullet points. It is designed to act like a person listening to you at 11pm when something is heavy on your chest — responding with extreme brevity, deep curiosity, and natural empathy.

## ✨ Core Features

- **Agentic Self-Correction**: The backend runs an internal "Draft -> Critic -> Refine" loop. An independent QA Critic evaluates Hearth's drafts against strict empathetic rules (no advice, extreme brevity, no psychoanalyzing) and forces rewrites until the response is perfectly humanized.
- **Modern Web UI**: A beautiful, premium chat interface featuring dark/light modes, micro-animations, and a calming glassmorphism aesthetic.
- **Fast Inference**: Powered by Groq (`llama-3.3-70b-versatile`) for instant, real-time responses.
- **Crisis Detection**: A parallel DistilBERT crisis classifier monitors for distress signals to ensure safe interactions.

---

## 📁 Project Structure

Following standard Python project architecture:

```text
Hearth/
├── src/
│   ├── api.py                 # Flask backend running the Agentic Evaluator Loop + UI serving
│   └── pipeline/              # Model scoring and evaluation pipelines
├── frontend/
│   └── hearth_chat_interface.html # The modern frontend chat interface
├── scripts/                   
│   ├── fetch_responses.py     # Script to test response generation
│   ├── mistral_api_finetune.py# Cloud fine-tuning via Mistral API
│   ├── prepare_data.py        # Data cleaning + split generation
│   ├── train.py               # Local LoRA fine-tuning (SFTTrainer)
│   └── crisis_classifier.py   # DistilBERT crisis detector
├── tests/                     # Test cases and runner scripts
├── docs/                      # Documentation and Agent Skills
├── data/                      # Raw and processed datasets
├── configs/                   # Hyperparameters (LoRA + training)
├── .env.example               # Environment template (API Keys)
└── requirements.txt
```

---

## 🚀 Quick Start (Running the App)

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set up API Keys

Create a `.env` file in the root directory and add your Groq API key:
```env
GROQ_API_KEY=gsk_your_key_here
```

### 3. Start the Backend Server

```bash
python src/api.py
```

### 4. Open the Interface
Navigate to [http://127.0.0.1:5000/](http://127.0.0.1:5000/) in your web browser. 
*(If you keep your terminal open, you can watch the Agentic Critic grading Hearth's drafts in real-time when you send a message!)*

---

## 🧠 The Agentic Loop Rules

The internal Critic ruthlessly enforces the following rules before allowing a message to reach the user:
1. **EXTREMELY SHORT**: Maximum 1-2 short sentences. No paragraphs.
2. **ZERO ADVICE**: No solutions, tips, or "fix-it" statements.
3. **NATURAL HUMAN FLOW**: No robotic or "point-manner" statements. 
4. **NO DRAMA**: Keep it grounded and raw. No cliches like "the weight of the world".
5. **NO AI APOLOGIES**: No "I'm sorry" or "As an AI".
6. **NO PSYCHOANALYZING**: If the user is vague (e.g. "I am lost"), Hearth must not assume why or analyze them. It must just provide a safe space ("That must be heavy. Tell me more.").

---

## 🔬 Fine-Tuning (Advanced)

If you want to bake the persona permanently into weights rather than relying entirely on the Agentic prompt loop, you have two options:

**Option A: Cloud Fine-Tuning (No GPU Required)**
1. Run `python scripts/prepare_data.py`
2. Run `python scripts/mistral_api_finetune.py` (Requires `MISTRAL_API_KEY`)

**Option B: Local LoRA Fine-Tuning (Requires 16GB+ VRAM)**
1. Run `python scripts/train.py`

---

*Hearth — Be curious. Be present. Be real.*
