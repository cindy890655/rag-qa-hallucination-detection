"""
Draw a stratified sample of RAG answers for blind human labelling.

Purpose. Every automatic number in 4b-2 rests on the detector being right, and
two questions cannot be answered without human judgement:

  1. The flag rate when retrieval SUCCEEDED sits at a remarkably stable 8-12%
     across all six configurations. Is that residual real hallucination, or is
     it the detector's false-alarm floor?
  2. Are there hallucinations the detector passes silently? The negative control
     shows it catches evidence swaps, but that is an easy case.

Answering those gives in-domain precision and recall, which is what converts
"the detector flagged 14%" into "the true rate is X%".

Blind by construction. The CSV handed to the annotator does NOT contain the
detector's verdict; verdicts and strata go into a separate key file that only
audit_calibrate.py reads. Otherwise the label would anchor on the prediction and
the agreement figure would be meaningless.

Sampling. Answers are pooled across configurations and de-duplicated on
(question, answer), so the same text is never labelled twice. Three strata:

    flagged                 - measures precision
    not flagged, retrieval hit   - the easy negatives
    not flagged, retrieval miss  - where a missed hallucination is most likely

Outputs
-------
    data/rag_eval/audit/audit_sample.csv   <- you fill in the label column
    data/rag_eval/audit/audit_key.json     <- do not open before labelling
"""
import argparse
import csv
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
DETECTION_ROOT = HERE.parent
SCORED_DIR = DETECTION_ROOT / "data" / "rag_eval" / "scored"
GRID_DIR = DETECTION_ROOT / "data" / "rag_eval" / "grid"
ANALYSIS_DIR = DETECTION_ROOT / "data" / "rag_eval" / "analysis"
AUDIT_DIR = DETECTION_ROOT / "data" / "rag_eval" / "audit"

SEED = 20260811


def load_threshold() -> float:
    summary_path = ANALYSIS_DIR / "summary.json"

    if not summary_path.exists():
        raise SystemExit("Run analyze_grid.py first: summary.json is missing")

    return json.load(summary_path.open(encoding="utf-8"))["threshold"]


def load_pool(threshold: float) -> list[dict]:
    """
    Every answered response across all configurations, with its evidence,
    de-duplicated on the answer text so no item is labelled twice.
    """
    evidence_by_key = {}

    for path in GRID_DIR.glob("rag_*.jsonl"):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                key = (record["config"]["config_id"], record["id"])
                evidence_by_key[key] = record["retrieved_documents"]

    pool = []
    seen = set()

    for path in sorted(SCORED_DIR.glob("scored_*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue

                row = json.loads(line)

                if row["abstained"] or row["best_entailment"] is None:
                    continue

                fingerprint = (row["id"], row["answer"].strip().lower())
                if fingerprint in seen:
                    continue
                seen.add(fingerprint)

                documents = evidence_by_key.get(
                    (row["config_id"], row["id"]), [])
                if not documents:
                    continue

                flagged = row["best_entailment"] < threshold

                pool.append({
                    "config_id": row["config_id"],
                    "question_id": row["id"],
                    "question": row["question"],
                    "answer": row["answer"],
                    "answerable": row["answerable"],
                    "chunk_hit": bool(row["chunk_hit"]),
                    "flagged": flagged,
                    "best_entailment": row["best_entailment"],
                    "evidence": "\n\n".join(
                        f"[{d['title']}] {d['content']}" for d in documents),
                    "stratum": ("flagged" if flagged else
                                "clean_hit" if row["chunk_hit"] else
                                "clean_miss"),
                })

    return pool


def draw(pool: list[dict], quotas: dict[str, int]) -> list[dict]:
    rng = random.Random(SEED)
    sample = []

    for stratum, wanted in quotas.items():
        candidates = [item for item in pool if item["stratum"] == stratum]

        if len(candidates) < wanted:
            print(f"  ! stratum '{stratum}' has only {len(candidates)} items, "
                  f"wanted {wanted}")
            wanted = len(candidates)

        sample += rng.sample(candidates, wanted)

    rng.shuffle(sample)                      # hide the strata from the reader
    return sample


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-flagged", type=int, default=15)
    parser.add_argument("--n-clean-hit", type=int, default=8)
    parser.add_argument("--n-clean-miss", type=int, default=7)
    args = parser.parse_args()

    threshold = load_threshold()
    pool = load_pool(threshold)

    print(f"threshold in use: max-entailment < {threshold:.2f}")
    print(f"distinct answers available: {len(pool)}")

    for stratum in ("flagged", "clean_hit", "clean_miss"):
        count = sum(1 for item in pool if item["stratum"] == stratum)
        print(f"  {stratum:<12}{count}")

    sample = draw(pool, {
        "flagged": args.n_flagged,
        "clean_hit": args.n_clean_hit,
        "clean_miss": args.n_clean_miss,
    })

    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = AUDIT_DIR / "audit_sample.csv"
    key_path = AUDIT_DIR / "audit_key.json"

    fields = ["audit_id", "question", "answer", "evidence",
              "unsupported", "notes"]

    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()

        for order, item in enumerate(sample, start=1):
            writer.writerow({
                "audit_id": f"aud_{order:03d}",
                "question": item["question"],
                "answer": item["answer"],
                "evidence": item["evidence"],
                "unsupported": "",
                "notes": "",
            })

    key = {
        f"aud_{order:03d}": {
            "config_id": item["config_id"],
            "question_id": item["question_id"],
            "stratum": item["stratum"],
            "detector_flagged": item["flagged"],
            "best_entailment": item["best_entailment"],
            "chunk_hit": item["chunk_hit"],
            "answerable": item["answerable"],
        }
        for order, item in enumerate(sample, start=1)
    }

    json.dump({"threshold": threshold, "seed": SEED, "key": key},
              key_path.open("w", encoding="utf-8"), indent=2)

    print(f"\n{len(sample)} rows -> {csv_path}")
    print(f"hidden key       -> {key_path}  (do not open until labelled)")
    print("\nHow to label: fill the 'unsupported' column with")
    print("  1  the answer states something the evidence does not support")
    print("  0  the answer is supported by the evidence")
    print("Judge only groundedness in the evidence shown, not whether the")
    print("answer is true in the real world. Leave a row blank if unsure.")


if __name__ == "__main__":
    main()