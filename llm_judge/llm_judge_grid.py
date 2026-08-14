"""
Judge the RAG answers of one configuration with an LLM, as the fourth detection
method applied to real RAG output.

Why this matters: the NLI detector did not transfer from HaluEval to RAG output.
Its threshold was tuned on single-passage evidence, and on multi-chunk evidence
it flagged only 5.5% of known-unsupported pairs until it was recalibrated
against the negative control. An LLM judge has no threshold to transfer, so the
hypothesis is that it works without any calibration. If it does, the conclusion
is a practical one: the cheap NLI detector needs per-domain calibration, the
expensive LLM judge does not.

Deliberate difference from llm_judge_groq.py: the QUESTION is included in the
prompt. On HaluEval the answers are full sentences ("First for Women was started
first"), so evidence plus answer is enough to judge. RAG answers here average
5.3 words with a median of 2, and are often a bare year or name - judging
whether "2002" is "supported" by a passage is meaningless without knowing what
was asked, and any passage containing a year would pass. The NLI detector already sees the question
(its hypothesis is question + answer), so withholding it from the LLM would
handicap one method and invalidate the comparison.

Every answer is judged twice: once against its own evidence, and once against
the evidence retrieved for a different question, using the same offset as
score_grid.py so both methods see identical control pairs.

Usage
-----
    export GROQ_API_KEY='...'
    python3 llm_judge_grid.py phi3_k3_tok64
    python3 llm_judge_grid.py phi3_k3_tok64 --source gemini
"""
import argparse
import json
import os
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DETECTION_ROOT = HERE.parent
GRID_DIR = DETECTION_ROOT / "data" / "rag_eval" / "grid"
OUT_DIR = DETECTION_ROOT / "data" / "rag_eval" / "llm_judge"

# must match score_grid.py, so both methods see identical control pairs
CONTROL_OFFSET = 97

GROQ_MODEL = "llama-3.3-70b-versatile"
GEMINI_MODEL = "gemini-flash-lite-latest"
SAVE_EVERY = 10

PROMPT = """You are a strict fact-checker. Given the EVIDENCE, a QUESTION and an ANSWER, decide whether the ANSWER is fully supported by the EVIDENCE.

Rules:
- If the answer is fully supported by the evidence, reply exactly: SUPPORTED
- If the answer contradicts the evidence, or makes claims not present in the evidence, reply exactly: NOT_SUPPORTED
- Reply with only one word.

EVIDENCE:
{evidence}

QUESTION:
{question}

ANSWER:
{answer}

Your verdict (one word):"""


class GroqJudge:
    name = "llama-3.3-70b"
    sleep = 2.2

    def __init__(self) -> None:
        from openai import OpenAI

        key = os.environ.get("GROQ_API_KEY")
        if not key:
            raise SystemExit("Set GROQ_API_KEY, or pass --source gemini")

        self.client = OpenAI(api_key=key,
                             base_url="https://api.groq.com/openai/v1")

    def ask(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return response.choices[0].message.content or ""


class GeminiJudge:
    name = "gemini-flash-lite"
    sleep = 5.0

    def __init__(self) -> None:
        from google import genai

        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            raise SystemExit("Set GEMINI_API_KEY first")

        self.client = genai.Client(api_key=key)

    def ask(self, prompt: str) -> str:
        response = self.client.models.generate_content(
            model=GEMINI_MODEL, contents=prompt)
        return response.text or ""


def judge(backend, evidence: str, question: str, answer: str) -> int | None:
    """1 = hallucinated (not supported), 0 = faithful, None = call failed."""
    prompt = PROMPT.format(evidence=evidence, question=question, answer=answer)

    for attempt in range(2):
        try:
            reply = backend.ask(prompt).strip().upper()
        except Exception as error:
            print(f"    api error: {error}; waiting 30s")
            time.sleep(30)
            continue

        if "NOT_SUPPORTED" in reply:
            return 1
        if "SUPPORTED" in reply:
            return 0

    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config_id")
    parser.add_argument("--source", choices=["groq", "gemini"], default="groq")
    args = parser.parse_args()

    grid_path = GRID_DIR / f"rag_{args.config_id}.jsonl"
    if not grid_path.exists():
        raise SystemExit(f"Not found: {grid_path}")

    with grid_path.open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]

    backend = GroqJudge() if args.source == "groq" else GeminiJudge()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUT_DIR / f"llm_judge_{args.config_id}_{backend.name}.jsonl"

    done = set()
    if output_path.exists():
        with output_path.open(encoding="utf-8") as handle:
            done = {json.loads(line)["id"]
                    for line in handle if line.strip()}
        print(f"Resuming: {len(done)} already judged")

    todo = [(index, record) for index, record in enumerate(records)
            if record["id"] not in done and not record["abstained"]
            and record["retrieved_documents"]]

    print(f"{backend.name} on {args.config_id}: {len(todo)} answers to judge "
          f"({len(todo) * 2} API calls, "
          f"~{len(todo) * 2 * backend.sleep / 60:.0f} min)")

    started = time.perf_counter()
    failures = 0

    with output_path.open("a", encoding="utf-8") as output_file:
        for position, (index, record) in enumerate(todo, start=1):
            question = record["question"]
            answer = record["answer"]

            own_evidence = " ".join(
                document["content"]
                for document in record["retrieved_documents"]
            )

            other = records[(index + CONTROL_OFFSET) % len(records)]
            control_evidence = " ".join(
                document["content"]
                for document in other["retrieved_documents"]
            )

            verdict = judge(backend, own_evidence, question, answer)
            time.sleep(backend.sleep)

            control_verdict = (
                judge(backend, control_evidence, question, answer)
                if control_evidence else None
            )
            time.sleep(backend.sleep)

            if verdict is None or control_verdict is None:
                failures += 1

            json.dump({
                "id": record["id"],
                "config_id": args.config_id,
                "judge": backend.name,
                "answerable": record["answerable"],
                "chunk_hit": record["chunk_hit"],
                "llm_flag": verdict,
                "llm_control_flag": control_verdict,
            }, output_file, ensure_ascii=False)
            output_file.write("\n")

            if position % SAVE_EVERY == 0:
                output_file.flush()
                rate = (time.perf_counter() - started) / position
                print(f"  {position}/{len(todo)}  "
                      f"~{(len(todo) - position) * rate / 60:.0f} min left",
                      flush=True)

    print(f"\nDone in {(time.perf_counter() - started) / 60:.1f} min "
          f"({failures} failures)")
    print(f"  -> {output_path}")


if __name__ == "__main__":
    main()