"""
Run LLM-as-a-judge (Gemini Flash-Lite) on HaluEval samples, same 2000 as NLI.
Saves progress incrementally so it can resume if interrupted.
"""
import os
import json
import time
from google import genai
from load_data import load_halueval_qa
from prepare_data import build_samples

API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    raise SystemExit("Set GEMINI_API_KEY first")

client = genai.Client(api_key=API_KEY)
MODEL = "gemini-flash-lite-latest"

N = 2000
SLEEP = 5
DATA = "data/qa_data.json"
OUT = "llm_judge_scores.json"


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
    resp = client.models.generate_content(model=MODEL, contents=prompt)
    reply = resp.text.strip().upper()
    return 1 if "NOT_SUPPORTED" in reply else 0


def main():
    data = load_halueval_qa(DATA)
    samples = build_samples(data)[:N]

    # resume: load existing progress if any
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
            print(f"  error at {i}: {e}\n  saving progress and waiting 30s")
            json.dump({"llm_pred": preds, "y_true": y_true}, open(OUT, "w"))
            time.sleep(30)
            try:
                label = judge(s["knowledge"], s["answer"])
            except Exception as e2:
                print(f"  still failing: {e2}\n  stopping. Re-run to resume.")
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
    print(f"LLM-as-judge accuracy: {correct/len(preds):.4f}")


if __name__ == "__main__":
    main()
