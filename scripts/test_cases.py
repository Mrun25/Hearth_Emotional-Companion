import os
import time
from dotenv import load_dotenv
from mistralai.client import Mistral

# Load environment variables
load_dotenv('.env')
api_key = os.environ.get("MISTRAL_API_KEY")

if not api_key:
    print("[ERROR] MISTRAL_API_KEY not found in .env")
    exit(1)

# Initialize Mistral client
client = Mistral(api_key=api_key)

# The Fumii System Prompt
FUMII_SYSTEM_PROMPT = (
    "You are Fumii, a 24-year-old friend texting. Rules for your voice:\n"
    "1. Write mostly in lowercase. Do not use perfect punctuation. Use run-on sentences sometimes.\n"
    "2. Use ellipses '...' for pauses and natural trailing thoughts.\n"
    "3. Use casual texting slang like 'yk' (you know), 'tbh', 'kinda', 'idk'.\n"
    "4. Ramble slightly and gently speculate before asking your question, just like a real person overthinking. "
    "(Example style: 'what makes you feel that... I mean you are aware of accomplishment so yk you have done enough maybe its something related to...').\n"
    "5. Do not use robotic empathy (e.g. 'I hear you', 'that sucks'). Be deeply human, slightly messy, and thoughtful.\n"
    "6. Keep it to 1-3 casual sentences."
)

# Test cases (User inputs)
test_cases = [
    "I'm feeling incredibly overwhelmed with work. Everything is piling up and I can't breathe.",
    "My best friend and I haven't spoken in days after a huge fight. I feel sick to my stomach.",
    "I accomplished a lot today, but I still feel empty inside for some reason.",
    "I'm really scared about the future. I don't know what I'm doing with my life.",
    "I just got off the phone with my mom and I'm crying. She never understands me."
]

print("=======================================================")
print("  FUMII MISTRAL API TEST (UN-FINETUNED BASE MODEL)")
print("=======================================================")
print("Model: ministral-8b-latest")
print("System Prompt applied to constrain behavior.")
print("=======================================================\n")

for i, test_case in enumerate(test_cases, 1):
    print(f"--- Test Case {i} ---")
    print(f"USER: {test_case}")
    
    try:
        # We use a standard chat model with the system prompt to simulate Fumii
        response = client.chat.complete(
            model="ministral-8b-latest",
            messages=[
                {"role": "system", "content": FUMII_SYSTEM_PROMPT},
                {"role": "user", "content": test_case}
            ]
        )
        
        fumii_reply = response.choices[0].message.content
        print(f"FUMII: {fumii_reply}\n")
    except Exception as e:
        print(f"[ERROR] API call failed: {e}\n")
        
    # Small delay to respect rate limits
    time.sleep(1)

print("Test complete.")
