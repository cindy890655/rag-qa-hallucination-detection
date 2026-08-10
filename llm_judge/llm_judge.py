"""
LLM-as-a-judge hallucination detector.
Given the knowledge (evidence) and the answer, ask Gemini whether the
answer is supported by the evidence. Returns 1 (hallucinated) or 0 (faithful).
"""
import os
import time
from google import genai

API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    raise SystemExit("Set GEMINI_API_KEY first: export GEMINI_API_KEY='your_key'")

client = genai.Client(api_key=API_KEY)
MODEL = "gemini-flash-lite-latest"


def judge(knowledge, answer):
    """
    Ask the LLM if the answer is supported by the knowledge.
    Returns (label, raw_reply): label = 1 if hallucinated, 0 if faithful.
    """
    prompt = f"""You are a strict fact-checker. Given the EVIDENCE and an ANSWER, decide whether the ANSWER is fully supported by the EVIDENCE.

Rules:
- If the answer is fully supported by the evidence, reply exactly: SUPPORTED
- If the answer contradicts the evidence, or makes claims not present in the evidence, reply exactly: NOT_SUPPORTED
- Reply with only one word: SUPPORTED or NOT_SUPPORTED

EVIDENCE:
{knowledge}

ANSWER:
{answer}

Your verdict (one word):"""

    resp = client.models.generate_content(model=MODEL, contents=prompt)
    reply = resp.text.strip().upper()
    # NOT_SUPPORTED => hallucinated (1); SUPPORTED => faithful (0)
    label = 1 if "NOT_SUPPORTED" in reply else 0
    return label, reply


# quick test on a few hand-made examples
if __name__ == "__main__":
    tests = [
        # (knowledge, answer, expected_label)
        ("Python was created by Guido van Rossum and released in 1991.",
         "Python was created by Guido van Rossum.", 0),   # supported
        ("Python was created by Guido van Rossum and released in 1991.",
         "Python was created by Elon Musk.", 1),          # not supported
        ("The Great Wall of China is over 13,000 miles long.",
         "The Great Wall of China is about 100 miles long.", 1),  # contradicts
    ]
    for kn, ans, expected in tests:
        label, reply = judge(kn, ans)
        mark = "OK" if label == expected else "WRONG"
        print(f"[{mark}] expected={expected} got={label} reply='{reply}'  | answer: {ans}")
        time.sleep(1)   # be gentle on rate limit