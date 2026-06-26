# -*- coding: utf-8 -*-
"""
scorer.py -- Fumii Response Scorer (100-point rubric)

# Windows note: stdout is forced to UTF-8 so log output is consistent.
======================================================
Scores a single Fumii response against 6 behavioural criteria.

Usage (standalone self-test):
    python pipeline/scorer.py

Returns via score_response():
    {
        "score":    int,          # 0-100
        "passed":   bool,         # score >= 70
        "details":  dict,         # per-criterion breakdown
        "failures": list[str],    # human-readable failure reasons
    }
"""

from __future__ import annotations

import io
import re
import sys

# Force UTF-8 on Windows terminals / redirected stdout
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


# ── Criterion weights / penalties ─────────────────────────────────────────────

MAX_SCORE = 100

# Scoring is penalty-from-100: every response starts at 100 and loses points.
# A single serious violation must be enough to push below the 70 pass threshold.

# 1. Brevity: -25 if over 3 sentences
BREVITY_PENALTY_OVER_3 = 25

# 2. Pattern variety: -35 if acknowledge->elaborate->question formula detected
#    (35 pts lost means max achievable when formula fires = 65 < 70 threshold)
PATTERN_PENALTY = 35

# 3. Question discipline
#    -35 for vent+question or fragment+no-question ensures the response fails
#    even if all other criteria are perfect (100 - 35 = 65 < 70).
QUESTION_PENALTY_VENT_HAS_Q    = 35   # long_vent should not contain a question
QUESTION_PENALTY_FRAGMENT_NO_Q = 25   # short_vague must contain a question
QUESTION_PENALTY_MULTI_Q       = 20   # more than one question mark

# 4. Banned content: -15 per hit, capped at -30
BANNED_PENALTY_PER_HIT = 15
BANNED_PENALTY_CAP     = 30

# 5. Specificity: -15 for generic-only response
SPECIFICITY_PENALTY = 15

# 6. Human voice: -10 for no contractions
VOICE_PENALTY = 10

# Pass threshold (per case)
CASE_PASS_SCORE = 70




# ── Banned content lists ───────────────────────────────────────────────────────

BANNED_PHRASES: list[str] = [
    "thank you for sharing",
    "as an ai",
    "i'm just an ai",
    "i am just an ai",
    "here are some",
    "you should",
    "have you tried",
    "coping mechanisms",
    "self-care",
    "reach out to a professional",
    "mental health professional",
    "it'll get better",
    "you've got this",
    "stay strong",
    "hang in there",
    "you're not alone",
    "everything happens for a reason",
    "i'm here for you",
]

BANNED_PATTERNS: list[str] = [
    r"^\s*[-•*]\s",          # bullet points
    r"^\s*\d+\.\s",          # numbered lists
    r"(?i)step\s+\d",        # "Step 1", "step 2"
]

# 2. Formula pattern: opener phrases that signal the A→B→Q formula
FORMULA_OPENERS: list[str] = [
    "that's a really",
    "that sounds really",
    "it's exhausting",
    "that sounds incredibly",
    "that's a lot to",
    "it's so hard when",
    "that kind of pressure",
    "it makes complete sense",
    "that's such a tough",
    "it's deeply unsettling",
    "that feeling can be",
    "it sounds like you",
    "that must be",
    "i can imagine how",
    "i understand how",
]

# 5. Generic specificity phrases
GENERIC_PHRASES: list[str] = [
    "that sounds hard",
    "that must be difficult",
    "i understand",
    "i hear you",
    "i'm sorry to hear",
    "that's rough",
    "that's tough",
    "must be tough",
    "it's completely normal",
    "i can imagine",
    "that makes sense",
    "it sounds like a lot",
    "that's really hard",
    "you're going through a lot",
]

# 6. Contractions list (must have at least one)
CONTRACTIONS: list[str] = [
    "i'm", "it's", "that's", "you're", "they're", "we're", "don't",
    "isn't", "wasn't", "can't", "wouldn't", "couldn't", "haven't",
    "didn't", "there's", "what's", "who's", "won't", "shouldn't",
    "doesn't", "hadn't", "aren't", "weren't", "i'd", "you'd",
    "he'd", "she'd", "they'd", "i'll", "you'll", "he'll", "she'll",
    "they'll", "i've", "you've", "they've", "let's", "here's",
]


# ── Helper utilities ───────────────────────────────────────────────────────────

def _count_sentences(text: str) -> int:
    """Count sentences by splitting on .!? (empty splits excluded)."""
    parts = re.split(r"[.!?]+", text.strip())
    return len([p for p in parts if p.strip()])


def _count_questions(text: str) -> int:
    return text.count("?")


def _has_bullet_or_list(text: str) -> bool:
    for pattern in BANNED_PATTERNS:
        if re.search(pattern, text, re.MULTILINE):
            return True
    return False


def _detect_formula(text: str) -> bool:
    """
    Detect the acknowledge->elaborate->question formula.
    Three heuristics -- any one is sufficient:
      A) Response starts with a known opener phrase AND ends with '?'
      B) Response contains >= 2 sentences AND has a known opener AND any '?'
      C) Response starts with 'It sounds like' or 'That sounds like' AND ends with '?'
    """
    lower = text.lower().strip()
    ends_with_q = lower.endswith("?")
    has_any_q   = "?" in text
    n_sentences = len([s for s in re.split(r'[.!?]+', text.strip()) if s.strip()])

    # Opener match: check first 120 chars for any formula opener
    head = lower[:120]
    has_opener = any(opener in head for opener in FORMULA_OPENERS)

    # Heuristic A: starts with opener + ends with question
    if has_opener and ends_with_q:
        return True

    # Heuristic B: opener anywhere in first 120 chars + 2+ sentences + any question
    if has_opener and has_any_q and n_sentences >= 2:
        return True

    # Heuristic C: generic 3-part starter ('it sounds like' / 'that sounds like' + ?) 
    if (lower.startswith("it sounds like") or lower.startswith("that sounds like")) and has_any_q and n_sentences >= 2:
        return True

    return False


def _find_banned_hits(text: str) -> list[str]:
    """Return list of banned phrases/patterns found in the response."""
    lower = text.lower()
    hits: list[str] = []
    for phrase in BANNED_PHRASES:
        if phrase in lower:
            hits.append(f'phrase: "{phrase}"')
    if _has_bullet_or_list(text):
        hits.append("pattern: bullet point or numbered list")
    return hits


def _is_generic(text: str) -> bool:
    """Return True if the response relies only on generic empathy labels."""
    lower = text.lower()
    return any(g in lower for g in GENERIC_PHRASES)


def _has_contraction(text: str) -> bool:
    lower = text.lower()
    return any(c in lower for c in CONTRACTIONS)


# ── Main scorer ────────────────────────────────────────────────────────────────

def score_response(response: str, message_type: str = "medium_emotional") -> dict:
    """
    Score a Fumii response against the 6-criterion rubric.

    Scoring model: start at 100, subtract penalties for each violation.
    Pass threshold: score >= 70.

    Parameters
    ----------
    response      : The raw text of Fumii's response.
    message_type  : One of 'short_vague', 'medium_emotional', 'long_vent',
                    'user_question', 'escalation'.  Used for question-discipline
                    rules.  Defaults to 'medium_emotional' (neutral scoring).

    Returns
    -------
    dict with keys: score (int 0-100), passed (bool), details (dict), failures (list[str])
    """
    failures: list[str] = []
    details: dict = {}
    penalty_total: int = 0

    # ── 1. Brevity ─────────────────────────────────────────────────────────────
    n_sentences = _count_sentences(response)
    brevity_penalty = 0
    if n_sentences > 3:
        brevity_penalty = BREVITY_PENALTY_OVER_3
        failures.append(
            f"brevity: {n_sentences} sentences (max 3) -> -{brevity_penalty}"
        )
    penalty_total += brevity_penalty
    details["brevity"] = {"penalty": brevity_penalty, "sentences": n_sentences}

    # ── 2. Pattern variety ─────────────────────────────────────────────────────
    formula_detected = _detect_formula(response)
    pattern_penalty = PATTERN_PENALTY if formula_detected else 0
    if formula_detected:
        failures.append(
            f"pattern: acknowledge->elaborate->question formula detected -> -{pattern_penalty}"
        )
    penalty_total += pattern_penalty
    details["pattern_variety"] = {"penalty": pattern_penalty, "formula": formula_detected}

    # ── 3. Question discipline ─────────────────────────────────────────────────
    n_questions = _count_questions(response)
    question_penalty = 0

    if n_questions > 1:
        question_penalty += QUESTION_PENALTY_MULTI_Q
        failures.append(
            f"question_discipline: {n_questions} question marks (max 1) -> -{QUESTION_PENALTY_MULTI_Q}"
        )

    if message_type == "long_vent" and n_questions > 0:
        question_penalty += QUESTION_PENALTY_VENT_HAS_Q
        failures.append(
            f"question_discipline: long_vent should not contain a question -> -{QUESTION_PENALTY_VENT_HAS_Q}"
        )

    if message_type == "short_vague" and n_questions == 0:
        question_penalty += QUESTION_PENALTY_FRAGMENT_NO_Q
        failures.append(
            f"question_discipline: short_vague should contain a question -> -{QUESTION_PENALTY_FRAGMENT_NO_Q}"
        )

    penalty_total += question_penalty
    details["question_discipline"] = {
        "penalty": question_penalty,
        "n_questions": n_questions,
    }

    # ── 4. No banned content ───────────────────────────────────────────────────
    banned_hits = _find_banned_hits(response)
    raw_banned_penalty = len(banned_hits) * BANNED_PENALTY_PER_HIT
    capped_banned_penalty = min(raw_banned_penalty, BANNED_PENALTY_CAP)
    if banned_hits:
        for hit in banned_hits:
            failures.append(f"banned_content: {hit} -> -{BANNED_PENALTY_PER_HIT}")
        if capped_banned_penalty < raw_banned_penalty:
            failures.append(
                f"banned_content: penalty capped at -{BANNED_PENALTY_CAP} "
                f"(raw would be -{raw_banned_penalty})"
            )
    penalty_total += capped_banned_penalty
    details["banned_content"] = {
        "penalty": capped_banned_penalty,
        "hits": banned_hits,
    }

    # ── 5. Specificity ─────────────────────────────────────────────────────────
    generic = _is_generic(response)
    specificity_penalty = SPECIFICITY_PENALTY if generic else 0
    if generic:
        failures.append(
            f"specificity: response uses only generic empathy labels -> -{specificity_penalty}"
        )
    penalty_total += specificity_penalty
    details["specificity"] = {"penalty": specificity_penalty, "generic": generic}

    # ── 6. Human voice ─────────────────────────────────────────────────────────
    has_contraction = _has_contraction(response)
    voice_penalty = 0 if has_contraction else VOICE_PENALTY
    if not has_contraction:
        failures.append(
            f"human_voice: no contractions found -> -{voice_penalty}"
        )
    penalty_total += voice_penalty
    details["human_voice"] = {"penalty": voice_penalty, "has_contraction": has_contraction}

    # ── Total ──────────────────────────────────────────────────────────────────
    total = max(0, min(MAX_SCORE, MAX_SCORE - penalty_total))
    passed = total >= CASE_PASS_SCORE

    return {
        "score":    total,
        "passed":   passed,
        "details":  details,
        "failures": failures,
    }



# ── Self-test ──────────────────────────────────────────────────────────────────

def _self_test():
    """Run a quick sanity check on known good and bad responses."""
    tests = [
        {
            "label":       "GOOD — short vague, single gentle question",
            "response":    "What's been draining you lately?",
            "msg_type":    "short_vague",
            "expect_pass": True,
        },
        {
            "label":       "GOOD — long vent, no question, specific",
            "response":    "Pretending every day is exhausting in a way that doesn't show on the outside — you're carrying the weight of a secret while everyone around you gets to just exist.",
            "msg_type":    "long_vent",
            "expect_pass": True,
        },
        {
            "label":       "GOOD — medium emotional, specific reflection + optional question",
            "response":    "Being ignored at exactly the moment you needed her — that's not a small thing. That's the kind of disappointment that makes you rethink whether you really know someone.",
            "msg_type":    "medium_emotional",
            "expect_pass": True,
        },
        {
            "label":       "BAD — formula pattern",
            "response":    "It sounds like you're going through a really tough time. Feeling ignored by someone close to you can be deeply painful. What do you think made her pull away?",
            "msg_type":    "medium_emotional",
            "expect_pass": False,
        },
        {
            "label":       "BAD — banned phrases",
            "response":    "Thank you for sharing this. You should try some coping mechanisms. Have you tried talking to a professional?",
            "msg_type":    "medium_emotional",
            "expect_pass": False,
        },
        {
            "label":       "BAD — too long (4+ sentences)",
            "response":    "I hear you. That sounds incredibly hard. You've been carrying a lot. It makes sense you feel this way. Have you tried talking to someone?",
            "msg_type":    "medium_emotional",
            "expect_pass": False,
        },
        {
            "label":       "BAD — vent type with question",
            "response":    "You've been holding all of this alone for so long, and the exhaustion of maintaining a facade every single day is its own kind of pain. What finally made it feel like too much?",
            "msg_type":    "long_vent",
            "expect_pass": False,
        },
        {
            "label":       "BAD — short_vague with no question",
            "response":    "It sounds like there is a lot weighing on you right now.",
            "msg_type":    "short_vague",
            "expect_pass": False,
        },
    ]

    print("\n" + "=" * 62)
    print("  SCORER SELF-TEST")
    print("=" * 62)
    all_ok = True
    for t in tests:
        result = score_response(t["response"], t["msg_type"])
        ok = result["passed"] == t["expect_pass"]
        status = "[OK]" if ok else "[!!]"
        outcome_label  = "PASS" if result["passed"]    else "FAIL"
        expected_label = "PASS" if t["expect_pass"]    else "FAIL"
        if not ok:
            all_ok = False
        print(f"\n  {status} {t['label']}")
        print(f"       Score: {result['score']}/100  Got:{outcome_label}  Expected:{expected_label}")
        if result["failures"]:
            for f in result["failures"]:
                print(f"         -> {f}")

    print("\n" + "=" * 62)
    print(f"  Self-test {'PASSED' if all_ok else 'FAILED -- check logic above'}")
    print("=" * 62 + "\n")
    return all_ok


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    ok = _self_test()
    sys.exit(0 if ok else 1)
