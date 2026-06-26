import os
import json
import urllib.request
import sys
from dotenv import load_dotenv

# Force UTF-8 output
import io
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    print("GROQ_API_KEY not found in .env")
    sys.exit(1)

# Fallback system prompt if not available
FUMII_SYSTEM_PROMPT = (
    "You are Fumii, a 24-year-old emotional companion. You are not a therapist, "
    "not a chatbot, not an assistant. You are the person someone talks to at 11pm "
    "when something is sitting heavy on their chest. "
    "3 sentences maximum. No advice. No bullet points. Use contractions always."
)

def generate_response(user_msg):
    url = "https://api.groq.com/openai/v1/chat/completions"
    data = {
        "model": "llama3-8b-8192",
        "messages": [
            {"role": "system", "content": FUMII_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg}
        ],
        "temperature": 0.75,
        "max_tokens": 150
    }
    
    req = urllib.request.Request(
        url,
        data=json.dumps(data).encode('utf-8'),
        headers={
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key}'
        }
    )
    
    try:
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode('utf-8'))
            return result['choices'][0]['message']['content']
    except Exception as e:
        return f"Error: {e}"

def main():
    try:
        with open("pipeline/eval_cases.json", "r", encoding="utf-8") as f:
            cases = json.load(f)
    except Exception as e:
        print(f"Error reading eval_cases.json: {e}")
        return

    print("Fetching actual answers for the first 5 testcases using Groq API (Llama 3)...\n")
    for case in cases[:5]:
        print("="*60)
        print(f"Testcase {case['id']} [{case['type']}]")
        print(f"User Message: {case['message']}")
        
        response = generate_response(case['message'])
        print(f"\nFumii Response: {response}")
        print("="*60 + "\n")

if __name__ == "__main__":
    main()
