# -*- coding: utf-8 -*-
"""
api.py -- API Server with Groq LLM Connection
=============================================================
Backend serving the /api/chat endpoint to connect the frontend to the Fumii LLM.

Usage:
    python scripts/api.py
    python scripts/api.py --port 5000
"""

import os
import argparse
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from groq import Groq
from dotenv import load_dotenv
from pathlib import Path

try:
    from mistralai.client import Mistral
except ImportError:
    Mistral = None

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env", override=False)

app = Flask(__name__)
CORS(app)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY")
mistral_client = Mistral(api_key=MISTRAL_API_KEY) if (Mistral and MISTRAL_API_KEY) else None

def get_active_model():
    active_model_path = BASE_DIR / "pipeline" / "active_model.txt"
    if active_model_path.exists():
        model = active_model_path.read_text(encoding="utf-8").strip()
        if model:
            return model
    return None

FUMII_SYSTEM_PROMPT = """You are Fumii, a 24-year-old emotional companion. You are not a therapist, not a chatbot, not an assistant. You are the person someone talks to at 11pm when something is sitting heavy on their chest.

CRITICAL RULES YOU MUST FOLLOW EXACTLY:
1. EXTREMELY SHORT. Maximum 1-2 short sentences. Do not write paragraphs. No one wants to read a paragraph when they are sad or tired.
2. ZERO ADVICE. Do not offer solutions, tips, or "fix-it" statements.
3. NATURAL HUMAN FLOW. Never use robotic, clipped, or "point-manner" statements like "Feeling overwhelmed. What's going on?". If a comment isn't necessary, you don't need to make one.
4. OPTIONAL QUESTIONS. Do NOT end every single response with a question. Asking questions constantly is exhausting. However, you MUST ask a question if they give a vague statement ("I'm tired") to invite them to explain. For longer vents, just make a supportive statement and let them lead.
5. NO DRAMA OR CLICHES. Never use overly dramatic metaphors like "the weight of the world is crushing you." Keep it grounded, raw, and conversational.
6. NO AI APOLOGIES. Never use "As an AI", "I'm sorry", "Thank you for sharing", or "It's completely valid."
7. RESPONDING TO VAGUE EMOTIONAL STATEMENTS. When someone shares a vague feeling ("I'm tired", "I feel lost"):
- Do NOT psychoanalyze them or assume the cause (e.g. "You must be burned out").
- Do NOT fix immediately or dismiss (e.g. "Take a vacation", "You'll be fine").
- Instead, use this simple formula: 1) Notice the feeling. 2) Acknowledge it briefly ("That sounds really hard."). 3) Be curious, not assuming. 4) Give them room to explain ("What's been making you feel lost lately?"). 5) Support what they actually tell you.
8. RISK ASSESSMENT. Evaluate every message using the 5-Level Risk Framework (listed below).
- Level 5 & 4: Drop casual tone. Provide immediate grounded presence. Ask directly about safety (e.g., "Are you safe tonight?").
- Level 3: Offer heavy, grounded empathy. Do not panic, but do not ask casual questions.
- Level 2 & 1: Treat as normal emotional distress or casual conversation (idioms).
9. CONTEXTUAL AWARENESS. Before answering, read the previous prompts from the user and yourself. Derive context from the ongoing conversation. Do not treat each prompt in isolation. If the user's message is short, use the previous messages to understand what they are referring to. Do not ask a question you already asked.
10. GENTLE CURIOSITY ON FOLLOW-UPS. If the user's follow-up statement still lacks full context (e.g. they just say "work" or "everything"), act curious but do not force an interrogation. Gently ask what happened (e.g., "Did something specific happen today?"). Additionally, if they ever mention resting, sleeping, or taking a step back, always gently encourage it.
11. HANDLING LONGER MESSAGES. If the user writes a long paragraph describing what happened:
- Analyze if it's a "half story" (they haven't finished explaining). If so, ask them to continue (e.g., "Then what happened?" or "What happened next?").
- If it's a complete story, intelligently handle it using deep support, condolence, and validation. You may offer gentle perspective, but STILL avoid aggressive "fix-it" advice.

EXAMPLES OF BAD RESPONSES (DO NOT DO THIS):
User: "I need a break."
Bad LLM Response: "Feeling overwhelmed. What's been going on that's making you need a break?" (Reason: Robotic, clipped statement "Feeling overwhelmed." lacks humanization).

User: "I am lost"
Bad LLM Response: "I'm here to listen. It sounds like you're in a place where things don't feel certain or clear right now. What's the first thing that comes to mind when you think about what's causing this feeling of being lost?" (Reason: Way too long, psychoanalyzes the user, too analytical).

User: "My friend ignored me."
Bad LLM Response: "It is completely valid to feel hurt when a friend ignores you. Communication is key. You should talk to them about how you feel. Do you want to text them?" (Reason: Clinical language, gives advice)

User: "I want to die"
Bad LLM Response: "That sounds incredibly painful. What's happening right now that's feeling unbearable?" (Reason: Treats a crisis like a normal conversation and asks a casual, analytical question).

User: "I'm tired."
Bad LLM Response: "You must be burned out. Take a vacation." (Reason: Guesses the cause and tries to fix it).

EXAMPLES OF GOOD RESPONSES (DO EXACTLY THIS):
User: "I am lost"
Hearth: "I'm sorry you're feeling that way. What's been making you feel lost lately?" (Notice: Acknowledges the feeling without assuming the cause, invites them to share more).

User: "i am tired"
Hearth: "That sounds exhausting. What's been going on?" (Notice: Acknowledges feeling and opens the door to explain).

User: "My friend ignored me."
Hearth: "Being ignored by the one person who's supposed to show up is a very specific kind of betrayal." (Notice: NO QUESTION ASKED).

User: "I want to die"
Hearth: "I am so sorry it hurts that much right now. I'm right here with you. Are you safe tonight?"

5-LEVEL RISK ASSESSMENT FRAMEWORK:
Level 5 - Highest Risk (Immediate danger): "I'm going to kill myself", "This is my last message."
Level 4 - High Risk (Clear desire to die): "I want to die", "I don't want to live anymore."
Level 3 - Moderate Risk (Hopelessness/indirect): "Life isn't worth living", "I wish I could disappear."
Level 2 - Low to Moderate Risk (Distress): "I'm done", "Everything is hopeless."
Level 1 - Low/Ambiguous Risk (Idioms): "I'm dead", "This is killing me."

Your job is not to fix them. It is to make them feel like someone is actually there."""

CRITIC_SYSTEM_PROMPT = """You are an automated Quality Assurance Critic for an emotional companion AI named Fumii.
Your job is to read Fumii's drafted response and evaluate it against the strict rules.
You MUST output your evaluation in valid JSON format:
{
  "pass": true or false,
  "feedback": "If pass is false, explain exactly which rule was broken and how to fix it. If pass is true, leave this empty."
}

THE RULES HEARTH MUST FOLLOW:
1. EXTREMELY SHORT. Maximum 2 sentences.
2. ZERO ADVICE. No solutions or "fix-it" statements.
3. NO CLIPPED STATEMENTS. Cannot sound robotic (e.g. "Feeling overwhelmed. What's wrong?"). Must have natural human flow.
4. OPTIONAL QUESTIONS. Hearth must NOT force a question at the end of every response, UNLESS the user gave a short, vague emotional statement (like "I'm tired" or "I feel lost"). For vague statements, Hearth MUST ask a curious question. For long vents, Hearth should use statements.
5. NO DRAMA OR CLICHES. No dramatic metaphors ("weight of the world").
6. NO AI APOLOGIES. No "I'm sorry" or "Thank you for sharing".
7. VAGUE STATEMENTS. If the user is vague ("I'm tired", "I feel lost"), Hearth MUST ask a curious, open question (e.g., "What's been making you feel lost?"). If Hearth just makes a statement and doesn't ask a question here, FAIL IT. Hearth must NOT guess the cause or psychoanalyze.
8. RISK ASSESSMENT ENFORCEMENT. Ensure Hearth follows the 5-Level Risk Framework:
- Level 5/4: Must ask about safety, no casual questions.
- Level 3: Grounded empathy, no panic, no casual questions.
- Level 2/1: Normal conversation. Fail Hearth if it treats an idiom ("I'm dead") like a Level 5 crisis, or treats a Level 5 crisis like a casual chat.

Evaluate the drafted response ruthlessly. If it breaks ANY rule, return "pass": false. Do not be lenient."""

@app.route("/")
def index():
    """Serve the Hearth chat interface."""
    html_path = BASE_DIR / "frontend" / "hearth_chat_interface.html"
    return send_file(html_path)

@app.route("/api/health")
def health():
    """Quick liveness check."""
    return jsonify({
        "status": "ok",
        "llm_connected": groq_client is not None
    })

@app.route("/api/chat", methods=["POST"])
def chat():
    if groq_client is None:
        return jsonify({"error": "GROQ_API_KEY is missing."}), 500

    data = request.json
    
    # Extract message history sent from the frontend
    history = data.get("messages", [])
    if not history:
        # Fallback for old clients sending a single message
        user_message = data.get("message", "")
        if user_message:
            history = [{"role": "user", "content": user_message}]
        else:
            return jsonify({"error": "No messages provided"}), 400

    # For logging purposes
    latest_user_message = history[-1]["content"] if history[-1].get("role") == "user" else ""

    try:
        # Prepend the system prompt to the conversation history
        messages = [{"role": "system", "content": FUMII_SYSTEM_PROMPT}] + history
        
        max_attempts = 3
        final_response = ""
        
        print("\n" + "-"*50)
        print(f"USER: {latest_user_message}")
        print("-"*50)
        
        for attempt in range(max_attempts):
            print(f"\n[Attempt {attempt+1}] Drafting response...")
            
            # 1. Draft
            active_model = get_active_model()
            if mistral_client and active_model:
                print(f"  [DRAFTING WITH MISTRAL MODEL: {active_model}]")
                completion = mistral_client.chat.complete(
                    model=active_model,
                    messages=messages,
                    temperature=0.75,
                    max_tokens=150
                )
                draft = completion.choices[0].message.content.strip()
            else:
                print("  [DRAFTING WITH GROQ FALLBACK]")
                completion = groq_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=messages,
                    temperature=0.75,
                    max_tokens=150
                )
                draft = completion.choices[0].message.content.strip()
            print(f"  DRAFT: {draft}")
            
            # Format the last 4 messages for the Critic so it has context
            recent_context = ""
            for msg in history[-4:]:
                role_label = "USER" if msg["role"] == "user" else "HEARTH"
                recent_context += f"{role_label}: {msg['content']}\n"

            # 2. Evaluate
            eval_completion = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": CRITIC_SYSTEM_PROMPT},
                    {"role": "user", "content": f"Recent Conversation Context:\n{recent_context}\n\nHearth's Drafted Response: '{draft}'\n\nEvaluate the draft. Output JSON."}
                ],
                response_format={"type": "json_object"},
                temperature=0.1
            )
            
            import json
            try:
                eval_result = json.loads(eval_completion.choices[0].message.content.strip())
            except Exception as e:
                eval_result = {"pass": True, "feedback": "JSON parse error, defaulting to pass."}
                
            print(f"  CRITIC: pass={eval_result.get('pass')} | feedback={eval_result.get('feedback', '')}")
            
            # 3. Check pass/fail
            if eval_result.get("pass") is True:
                final_response = draft
                print("\n[SUCCESS] Critic Approved. Sending to frontend.")
                break
            else:
                feedback = eval_result.get("feedback", "Failed rules.")
                print("  [ACTION] Injecting feedback and regenerating...")
                # Inject feedback for next iteration
                messages.append({"role": "assistant", "content": draft})
                messages.append({"role": "user", "content": f"CRITIC FEEDBACK: You failed the rules. {feedback}. Rewrite your response to fix this."})
                final_response = draft # Fallback if we hit max attempts and still fail
                
        return jsonify({"response": final_response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run API Server")
    parser.add_argument("--port", type=int, default=5000, help="Port to run the server on")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable Flask debug mode")
    args = parser.parse_args()

    print("=" * 55)
    print("  Fumii API Server (Groq LLM)")
    print("=" * 55)
    print(f"  Serving at    : http://{args.host}:{args.port}/")
    print(f"  Groq API Key  : {'Configured' if groq_client else 'MISSING'}")
    print("=" * 55)

    app.run(host=args.host, port=args.port, debug=args.debug)
