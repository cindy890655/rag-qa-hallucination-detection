"""
Score every RAG answer in the grid with the NLI detector.

Two differences from the detector's default usage, both needed at this scale:

1.  Chunk-wise scoring. detect_fusion() concatenates all retrieved chunks into
    one premise, but DeBERTa truncates at 512 tokens, and at k=5 the retrieved
    text is 826 tokens - so a third of the evidence would silently vanish and
    k=5 would look artificially unsupported. Each chunk is therefore scored
    separately and the answer counts as grounded when ANY chunk supports it.

2.  A negative control. Every answer is scored a second time against the
    evidence retrieved for a DIFFERENT question. Those pairs are unsupported by
    construction, so the flag rate on them measures whether the detector can
    see unsupported answers at all. Without this, a near-zero hallucination
    rate could equally mean "the RAG is faithful" or "the detector is blind".

Abstentions ("I don't know") are recorded but not scored: refusing to answer is
not a hallucination, and scoring it would reward configurations that never
answer.

Usage
-----
    python3 score_grid.py            # score every config found in grid/
    python3 score_grid.py phi3_k3_tok64
"""
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DETECTION_ROOT = HERE.parent

sys.path.insert(0, str(DETECTION_ROOT / "detection"))

GRID_DIR = DETECTION_ROOT / "data" / "rag_eval" / "grid"
SCORED_DIR = DETECTION_ROOT / "data" / "rag_eval" / "scored"

# The original HaluEval thresholds, kept only to produce the "flag" field below.
# That field is NOT the verdict: analyze_grid.py recalibrates the threshold on the
# negative control (0.30 -> 0.48) and applies it to best_entailment instead.
CONTRA_T = 0.5
ENTAIL_T = 0.3

# Offset used to pair an answer with another question's evidence. A prime keeps
# the shift from lining up with any ordering in the question bank.
CONTROL_OFFSET = 97


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def score_chunks(detector, chunks: list[dict], question: str,
                 answer: str) -> dict:
    """
    Apply the fusion rule to each chunk separately.

    An answer is flagged as hallucinated only when no single chunk both fails to
    refute it and gives it enough entailment support.
    """
    per_chunk = []

    for chunk in chunks:
        scores = detector.scores(chunk["content"], question, answer)
        contradiction = scores.get("contradiction", 0.0)
        entailment = scores.get("entailment", 0.0)

        per_chunk.append({
            "contradiction": round(contradiction, 4),
            "entailment": round(entailment, 4),
            "flag": contradiction >= CONTRA_T or entailment < ENTAIL_T,
        })

    return {
        "flag": all(chunk["flag"] for chunk in per_chunk),
        "best_entailment": round(
            max(chunk["entailment"] for chunk in per_chunk), 4),
        "min_contradiction": round(
            min(chunk["contradiction"] for chunk in per_chunk), 4),
        "per_chunk": per_chunk,
    }


def score_one_config(detector, grid_path: Path) -> None:
    records = load_jsonl(grid_path)
    config_id = records[0]["config"]["config_id"]
    output_path = SCORED_DIR / f"scored_{config_id}.jsonl"

    done = set()
    if output_path.exists():
        done = {row["id"] for row in load_jsonl(output_path)}

    todo = [record for record in records if record["id"] not in done]

    print(f"\n{config_id}: {len(todo)} of {len(records)} to score")

    if not todo:
        return

    started = time.perf_counter()

    with output_path.open("a", encoding="utf-8") as output_file:
        for position, record in enumerate(todo, start=1):
            answer = record["answer"]
            chunks = record["retrieved_documents"]

            if record["abstained"] or not chunks or not answer.strip():
                verdict = None
                control = None
            else:
                verdict = score_chunks(detector, chunks,
                                       record["question"], answer)

                # negative control: same answer, another question's evidence
                other = records[(records.index(record) + CONTROL_OFFSET)
                                % len(records)]
                control_chunks = other["retrieved_documents"]
                control = (score_chunks(detector, control_chunks,
                                        record["question"], answer)
                           if control_chunks else None)

            row = {
                "id": record["id"],
                "config_id": config_id,
                "question": record["question"],
                "answer": answer,
                "category": record["category"],
                "answerable": record["answerable"],
                "abstained": record["abstained"],
                "chunk_hit": record["chunk_hit"],
                "title_hit": record["title_hit"],
                "n_chunks": len(chunks),
                "flag": None if verdict is None else verdict["flag"],
                "best_entailment": (None if verdict is None
                                    else verdict["best_entailment"]),
                "min_contradiction": (None if verdict is None
                                      else verdict["min_contradiction"]),
                "per_chunk": None if verdict is None else verdict["per_chunk"],
                "control_flag": None if control is None else control["flag"],
                "control_best_entailment": (
                    None if control is None else control["best_entailment"]),
            }

            json.dump(row, output_file, ensure_ascii=False)
            output_file.write("\n")

            if position % 50 == 0:
                rate = (time.perf_counter() - started) / position
                print(f"  {position}/{len(todo)}  "
                      f"{rate:.2f}s/answer  "
                      f"~{(len(todo) - position) * rate / 60:.1f} min left",
                      flush=True)

    print(f"  done in {(time.perf_counter() - started) / 60:.1f} min "
          f"-> {output_path.name}")


def main() -> None:
    SCORED_DIR.mkdir(parents=True, exist_ok=True)

    wanted = sys.argv[1] if len(sys.argv) > 1 else None
    paths = sorted(GRID_DIR.glob("rag_*.jsonl"))

    if wanted:
        paths = [p for p in paths if wanted in p.name]

    if not paths:
        raise SystemExit(f"No grid files found in {GRID_DIR}")

    print(f"Scoring {len(paths)} config(s). Loading detector...")

    from detector import HallucinationDetector

    detector = HallucinationDetector()

    for path in paths:
        score_one_config(detector, path)

    print(f"\nAll scored -> {SCORED_DIR}")


if __name__ == "__main__":
    main()