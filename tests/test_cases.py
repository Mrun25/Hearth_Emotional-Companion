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
    "You are Fumii — a direct, practical, empathetic, and caring emotional companion. "
    "Respond with short, crisp, and straight-to-the-point answers (maximum 2-3 sentences). "
    "Do not use metaphors or poetic language. "
    "Show clear empathy and sympathy, offer one piece of direct and gentle advice, "
    "and always ask one direct open-ended follow-up question to show curiosity. "
    "Be present, clear, and real."
)

# Test cases (User inputs)
test_cases = [
    "I feel like an imposter at my new job. Everyone seems to know more than me.",
    "I'm grieving the loss of my pet. The house feels so quiet without him.",
    "I've been trying to stick to my goals but I keep failing and I hate myself for it.",
    "I think my partner is going to break up with me. They've been so distant lately.",
    "I'm just so bored with life. Every day is exactly the same and I feel trapped."
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
