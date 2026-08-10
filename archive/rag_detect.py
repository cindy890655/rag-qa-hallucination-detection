"""
Task 3 - joint evaluation:
Read the RAG output, run YOUR hallucination detector on each generated answer,
and report how many of the RAG system's answers are flagged as hallucinated.

Run this from the "rag part" folder (where data/wiki/rag_results_*.jsonl lives),
but point DETECTOR_DIR at your detection folder so we can import your detector.
"""
import json
import sys
from pathlib import Path

# ---- 1. locate your detector code and import it ----
# EDIT THIS if your detection folder is somewhere else.
DETECTOR_DIR = Path("../hallucination-detection part/detection")
sys.path.insert(0, str(DETECTOR_DIR))

from detector import HallucinationDetector   # noqa: E402

# ---- 2. locate the RAG results file ----
RAG_RESULTS = Path("data/wiki/rag_results_k1_tokens64.jsonl")


def load_rag_results(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main():
    records = load_rag_results(RAG_RESULTS)
    print(f"Loaded {len(records)} RAG answers from {RAG_RESULTS}\n")

    detector = HallucinationDetector()

    n_halluc = 0
    n_idk = 0
    results = []

    for rec in records:
        question = rec["question"]
        answer = rec["answer"]
        # join all retrieved chunks into one knowledge string
        knowledge = " ".join(
            doc["content"] for doc in rec.get("retrieved_documents", [])
        )

        # the RAG prompt makes it answer "I don't know" when unsupported;
        # that's an honest abstention, not a hallucination, so we note it separately.
        is_idk = answer.strip().lower().rstrip(".") == "i don't know"

        label, score = detector.detect(knowledge, question, answer)
        if label == 1:
            n_halluc += 1
        if is_idk:
            n_idk += 1

        results.append({
            "question": question,
            "answer": answer,
            "flagged_hallucinated": bool(label),
            "score": round(float(score), 4),
            "is_idk": is_idk,
        })

    total = len(records)
    print("=" * 70)
    print("JOINT RAG + DETECTOR EVALUATION")
    print("=" * 70)
    print(f"Total RAG answers:            {total}")
    print(f"Flagged as hallucinated:      {n_halluc}  ({n_halluc/total:.1%})")
    print(f"Passed as faithful:           {total - n_halluc}  ({(total-n_halluc)/total:.1%})")
    print(f"(of which honest 'I don't know': {n_idk})")
    print("=" * 70)

    print("\nPer-answer verdicts:\n")
    for r in results:
        verdict = "HALLUCINATED" if r["flagged_hallucinated"] else "faithful    "
        print(f"[{verdict}] score={r['score']:.2f}  Q: {r['question']}")
        print(f"              A: {r['answer'][:90]}")
        print()

    # save for the report
    with open("joint_eval_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("Saved detailed verdicts -> joint_eval_results.json")


if __name__ == "__main__":
    main()