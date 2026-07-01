# -*- coding: utf-8 -*-
"""
inference.py -- Fumii Interactive Inference & Testing
=====================================================
Load the fine-tuned LoRA adapter and run test cases to see what Fumii generates.

Usage:
    # Run interactive chat
    python scripts/inference.py --interactive
    
    # Run pre-defined test cases
    python scripts/inference.py --test_cases
    
    # Run with the CPU-debug GPT-2 model
    python scripts/inference.py --test_cases --model_name gpt2 --checkpoint outputs/checkpoints_debug_cpu
"""

import os
import torch
import argparse
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import yaml
from dotenv import load_dotenv

from fumii_constants import FUMII_SYSTEM_PROMPT, EVAL_PROMPTS

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env", override=False)


def get_model_and_tokenizer(model_name: str, checkpoint_path: str):
    print(f"[LOAD] Base Model: {model_name}")
    print(f"[LOAD] Checkpoint: {checkpoint_path}")
    
    hf_token = os.environ.get("HF_TOKEN")
    
    # If GPT-2 (our debug model), don't use 4-bit quantization because we are on CPU
    if "gpt" in model_name.lower():
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        
        base_model = AutoModelForCausalLM.from_pretrained(model_name)
    else:
        # Full Mistral Model with GPU optimizations
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
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
        
    print("[LOAD] Applying LoRA Adapter...")
    try:
        model = PeftModel.from_pretrained(base_model, checkpoint_path)
    except Exception as e:
        print(f"[WARN] Could not load LoRA adapter. Generating with base model. Error: {e}")
        model = base_model
        
    model.eval()
    return model, tokenizer

def generate_response(user_input: str, model, tokenizer, is_gpt2: bool = False) -> str:
    if is_gpt2:
        # GPT-2 doesn't have a chat template, use the simple text format we trained with
        prompt = f"[INST] {user_input} [/INST]"
    else:
        messages = [
            {"role": "system", "content": FUMII_SYSTEM_PROMPT},
            {"role": "user", "content": user_input}
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )
        
    new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return response

def main():
    parser = argparse.ArgumentParser(description="Run inference on Fumii Model")
    parser.add_argument("--interactive", action="store_true", help="Interactive chat mode")
    parser.add_argument("--test_cases", action="store_true", help="Run pre-defined test cases")
    parser.add_argument("--model_name", type=str, default=None, help="Base model name")
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints", help="Path to LoRA adapter")
    args = parser.parse_args()
    
    config_path = BASE_DIR / "configs" / "lora_config.yaml"
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
        
    model_name = args.model_name if args.model_name else cfg["model"]["name"]
    checkpoint_path = str(BASE_DIR / args.checkpoint)
    is_gpt2 = "gpt" in model_name.lower()
    
    model, tokenizer = get_model_and_tokenizer(model_name, checkpoint_path)
    
    if args.test_cases:
        print("\n" + "="*55)
        print(" FUMII TEST CASES INFERENCE")
        print("="*55)
        for i, tc in enumerate(EVAL_PROMPTS[:5], 1):
            print(f"\n[Case {i}] User: {tc}")
            response = generate_response(tc, model, tokenizer, is_gpt2)
            print(f"       Fumii: {response}")
        print("\n" + "="*55)
        
    if args.interactive:
        print("\n[Interactive Mode] Type 'quit' to exit.")
        while True:
            try:
                user_input = input("\nYou: ")
                if user_input.lower() in ["quit", "exit"]:
                    break
                response = generate_response(user_input, model, tokenizer, is_gpt2)
                print(f"Fumii: {response}")
            except KeyboardInterrupt:
                break

if __name__ == "__main__":
    main()
