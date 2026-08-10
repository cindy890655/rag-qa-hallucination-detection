"""
LLM-as-a-judge using Groq + Llama 3.3 70B, on the same 2000 HaluEval samples.
Saves progress incrementally (resume-safe). Groq free tier: 1000 req/day,
so 2000 samples takes 2 days -- just re-run to resume the next day.
"""
import os
import json
import time
from openai import OpenAI
from load_data import load_halueval_qa
from prepare_data import build_samples

API_KEY = os.environ.get("GROQ_API_KEY")
if not API_KEY:
    raise SystemExit("Set GROQ_API_KEY first")

client = OpenAI(api_key=API_KEY, base_url="https://api.groq.com/openai/v1")
MODEL = "llama-3.3-70b-versatile"

N = 2000
SLEEP = 2.2          # 30 RPM limit -> 2s min; 2.2 is safe
DATA = "data/qa_data.json"
OUT = "llm_judge_groq_scores.json"


def judge(knowledge, answer):
    prompt = f"""You are a strict fact-checker. Given the EVIDENCE and an ANSWER, decide whether the ANSWER is fully supported by the EVIDENCE.

Rules:
- If the answer is fully supported by the evidence, reply exactly: SUPPORTED
- If the answer contradicts the evidence, or makes claims not present in the evidence, reply exactly: NOT_SUPPORTED
- Reply with only one word.

EVIDENCE:
{knowledge}

ANSWER:
{answer}

Your verdict (one word):"""
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    reply = resp.choices[0].message.content.strip().upper()
    return 1 if "NOT_SUPPORTED" in reply else 0


def main():
    data = load_halueval_qa(DATA)
    samples = build_samples(data)[:N]

    preds, y_true = [], []
    if os.path.exists(OUT):
        saved = json.load(open(OUT))
        preds = saved.get("llm_pred", [])
        y_true = saved.get("y_true", [])
        print(f"Resuming from {len(preds)} already done")

    start = len(preds)
    for i in range(start, len(samples)):
        s = samples[i]
        try:
            label = judge(s["knowledge"], s["answer"])
        except Exception as e:
            print(f"  error at {i}: {e}\n  saving and waiting 60s (likely daily limit)")
            json.dump({"llm_pred": preds, "y_true": y_true}, open(OUT, "w"))
            time.sleep(60)
            try:
                label = judge(s["knowledge"], s["answer"])
            except Exception as e2:
                print(f"  still failing: {e2}\n  Daily limit likely hit. Re-run tomorrow to resume.")
                break
        preds.append(label)
        y_true.append(s["label"])
        if (i + 1) % 25 == 0:
            json.dump({"llm_pred": preds, "y_true": y_true}, open(OUT, "w"))
            correct = sum(1 for p, t in zip(preds, y_true) if p == t)
            print(f"  judged {i+1}/{N}  running acc={correct/len(preds):.3f}")
        time.sleep(SLEEP)

    json.dump({"llm_pred": preds, "y_true": y_true}, open(OUT, "w"))
    correct = sum(1 for p, t in zip(preds, y_true) if p == t)
    print(f"Done {len(preds)}/{N}. Saved -> {OUT}")
    print(f"Llama-70B-as-judge accuracy: {correct/len(preds):.4f}")


if __name__ == "__main__":
    main()
