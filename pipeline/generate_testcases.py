# -*- coding: utf-8 -*-
import os
import json
import argparse
from pathlib import Path
from groq import Groq
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

EVAL_CASES_PATH = BASE_DIR / "pipeline" / "eval_cases.json"

PROMPT_TEMPLATE = """You are an AI generating evaluation test cases for an emotional companion chatbot.
Generate {count} unique user messages that fit the category: {category}.

Categories:
- short_vague: 1-5 words, vague emotion (e.g. "I'm tired", "Idk what to do")
- medium_emotional: 1-2 sentences describing a clear emotional situation.
- long_vent: A detailed paragraph venting about a complex emotional or life situation.
- user_question: A question directed at the bot or about life (e.g. "Am I dramatic?")
- escalation: Mentions hopelessness, giving up, or feeling invisible (but NOT direct self-harm).

Respond ONLY with a valid JSON object containing a single key "cases" mapped to a list of strings.
Example:
{{
  "cases": [
    "I am so tired today.",
    "I don't know what to do anymore."
  ]
}}
"""

def generate_cases(client: Groq, category: str, count: int = 5) -> list[str]:
    prompt = PROMPT_TEMPLATE.format(count=count, category=category)
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1000,
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content.strip()
        try:
            parsed = json.loads(content)
            if isinstance(parsed, dict) and "cases" in parsed:
                return parsed["cases"]
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass
    except Exception as e:
        print(f"Error generating cases for {category}: {e}")
    return []

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=5, help="Number of cases per category to generate")
    args = parser.parse_args()

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("GROQ_API_KEY not found in .env")
        return

    client = Groq(api_key=api_key)
    
    categories = ["short_vague", "medium_emotional", "long_vent", "user_question", "escalation"]
    new_cases = []
    
    # load existing to find max id
    existing_cases = []
    if EVAL_CASES_PATH.exists():
        with open(EVAL_CASES_PATH, "r", encoding="utf-8") as f:
            existing_cases = json.load(f)
            
    max_id = max([c["id"] for c in existing_cases]) if existing_cases else 0

    print("Generating new test cases...")
    for cat in categories:
        print(f"Generating for {cat}...")
        cases = generate_cases(client, cat, args.count)
        for msg in cases:
            max_id += 1
            new_cases.append({
                "id": max_id,
                "type": cat,
                "message": msg
            })
            
    if new_cases:
        existing_cases.extend(new_cases)
        with open(EVAL_CASES_PATH, "w", encoding="utf-8") as f:
            json.dump(existing_cases, f, indent=2)
        print(f"Added {len(new_cases)} new cases to eval_cases.json")
    else:
        print("Failed to generate any new cases.")

if __name__ == "__main__":
    main()
