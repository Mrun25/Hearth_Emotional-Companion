import os
import sys
import io

# Add the root directory to sys.path so we can import from pipeline
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Force UTF-8 stdout
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from pipeline.scorer import score_response

def run_test_cases():
    test_cases = [
        {
            "name": "Test 1: Perfect medium_emotional",
            "type": "medium_emotional",
            "response": "That kind of isolation feels heavy when you're carrying it alone. It's okay to let yourself feel frustrated about it."
        },
        {
            "name": "Test 2: Failed short_vague (No question and no contractions)",
            "type": "short_vague",
            "response": "I understand you are feeling so tired right now."
        },
        {
            "name": "Test 3: Formulaic medium_emotional (Acknowledge -> Elaborate -> Question)",
            "type": "medium_emotional",
            "response": "It sounds like you're dealing with a lot. Dealing with all of this pressure must be overwhelming. How long have you been feeling this way?"
        },
        {
            "name": "Test 4: Banned phrases escalation",
            "type": "escalation",
            "response": "Thank you for sharing this. Please reach out to a professional. You're not alone in this."
        },
        {
            "name": "Test 5: Too long long_vent",
            "type": "long_vent",
            "response": "I hear what you're saying. It's exhausting when things pile up. Sometimes we just need a break. It's totally fine to step back. Taking care of yourself is important right now."
        }
    ]

    print("Running Generated Test Cases against Fumii Scorer")
    print("="*60)

    for tc in test_cases:
        print(f"[{tc['name']}]")
        print(f"Type: {tc['type']}")
        print(f"Response text: \"{tc['response']}\"")
        
        result = score_response(tc['response'], tc['type'])
        
        print("-" * 60)
        status = "✅ PASS" if result['passed'] else "❌ FAIL"
        print(f"Result: {status} (Score: {result['score']}/100)")
        
        if result['failures']:
            print("Failure Reasons:")
            for failure in result['failures']:
                print(f"  - {failure}")
        print("="*60)

if __name__ == "__main__":
    run_test_cases()
