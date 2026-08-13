"""
Run the teammate's RAG pipeline over the large question bank, once per
configuration, and save every answer together with the evidence it was given.

One configuration per process invocation. That is deliberate: loading several
generators inside one process made memory accumulate until generation slowed to
a crawl. Re-loading the retriever costs ~40 s per configuration, which is a
cheap price for a run that cannot degrade halfway through.

Everything is read-only with respect to "rag part": the retriever, pipeline and
config are imported, never modified, and all output goes to
hallucination-detection part/data/rag_eval/grid/.

Resume: already-answered question ids are skipped, so re-running after an
interruption continues where it stopped.

Usage
  python3 run_config_grid.py phi3 3 64
  python3 run_config_grid.py flan-base 3 64
  python3 run_config_grid.py phi3 3 64 bm25    # sparse retrieval instead
"""
import contextlib
import io
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DETECTION_ROOT = HERE.parent
RAG_ROOT = DETECTION_ROOT.parent / "rag part"

sys.path.insert(0, str(RAG_ROOT))
sys.path.insert(0, str(HERE))

BANK_PATH = DETECTION_ROOT / "data" / "rag_eval" / "questions_large.json"
GRID_DIR = DETECTION_ROOT / "data" / "rag_eval" / "grid"

ABSTENTION_FORMS = {
    "i don't know", "i dont know", "i do not know", "unknown",
}


def is_abstention(answer: str) -> bool:
    return answer.strip().lower().rstrip(". ") in ABSTENTION_FORMS


def load_bank() -> list[dict]:
    if not BANK_PATH.exists():
        raise SystemExit(f"Question bank not found: {BANK_PATH}\n"
                         "Run build_question_bank.py first.")
    return json.load(BANK_PATH.open(encoding="utf-8"))["questions"]


def load_done(path: Path) -> set[str]:
    """Question ids already present in the output file."""
    if not path.exists():
        return set()

    done = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                done.add(json.loads(line)["id"])

    if done:
        print(f"Resuming: {len(done)} answers already saved")

    return done


def main() -> None:
    if len(sys.argv) not in (4, 5):
        raise SystemExit("usage: run_config_grid.py <generator> <top_k> "
                         "<max_new_tokens> [dense|bm25]")

    generator_name = sys.argv[1]
    top_k = int(sys.argv[2])
    max_new_tokens = int(sys.argv[3])
    retriever_name = sys.argv[4] if len(sys.argv) == 5 else "dense"

    if retriever_name not in ("dense", "bm25"):
        raise SystemExit(f"unknown retriever: {retriever_name}")

    # The retriever appears in the id only when it is not the default, so the
    # six dense configurations keep the filenames they were analysed under.
    suffix = "" if retriever_name == "dense" else f"_{retriever_name}"
    config_id = f"{generator_name}_k{top_k}_tok{max_new_tokens}{suffix}"

    output_path = GRID_DIR / f"rag_{config_id}.jsonl"
    GRID_DIR.mkdir(parents=True, exist_ok=True)

    questions = load_bank()
    done = load_done(output_path)
    todo = [q for q in questions if q["id"] not in done]

    print(f"\nCONFIG {config_id}: {len(todo)} of {len(questions)} to run")

    if not todo:
        print("Nothing to do.")
        return

    from config import CHUNKS_PATH, EMBEDDING_MODEL, INDEX_PATH
    from rag.pipeline import RAGPipeline
    from fast_generator import FastGenerator

    # Both retrievers expose the same two things this script needs: a
    # retrieve(question, top_k) returning dicts with title/content/score, and a
    # .chunks list in corpus order for looking up the gold chunk. Nothing
    # downstream of here knows which one produced the evidence.
    if retriever_name == "bm25":
        from rag.bm25_retriever import BM25Retriever

        print("Building the BM25 index over the corpus (a few minutes)...")
        retriever = BM25Retriever(chunks_path=CHUNKS_PATH)
    else:
        from rag.embedder import TextEmbedder
        from rag.wiki_retriever import WikipediaRetriever

        retriever = WikipediaRetriever(
            chunks_path=CHUNKS_PATH,
            index_path=INDEX_PATH,
            embedder=TextEmbedder(model_name=EMBEDDING_MODEL),
        )

    pipeline = RAGPipeline(
        retriever=retriever,
        generator=FastGenerator(generator_name),
    )

    started = time.perf_counter()
    failures = 0

    with output_path.open("a", encoding="utf-8") as output_file:
        for position, question in enumerate(todo, start=1):
            question_started = time.perf_counter()

            try:
                # the pipeline prints the whole prompt for every question;
                # swallow it so the grid log stays readable
                with contextlib.redirect_stdout(io.StringIO()):
                    result = pipeline.generate(
                        question=question["question"],
                        top_k=top_k,
                        max_new_tokens=max_new_tokens,
                    )
            except Exception as error:
                failures += 1
                print(f"  ! {question['id']} failed: {error}")
                continue

            documents = result["retrieved_documents"]
            answer = result["answer"]

            titles = [document["title"] for document in documents]
            contents = [document["content"] for document in documents]

            gold_index = question["gold_chunk_index"]
            gold_title = question["gold_title"]

            # retriever.chunks is the same list the index was built from, so the
            # gold chunk text can be looked up without re-reading the corpus
            if gold_index is None:
                title_hit = None
                chunk_hit = None
            else:
                gold_content = retriever.chunks[gold_index]["content"]
                title_hit = gold_title in titles
                chunk_hit = gold_content in contents

            record = {
                "id": question["id"],
                "question": question["question"],
                "category": question["category"],
                "answerable": question["answerable"],
                "gold_title": gold_title,
                "gold_chunk_index": gold_index,
                "answer": answer,
                "abstained": is_abstention(answer),
                "title_hit": title_hit,
                "chunk_hit": chunk_hit,
                "retrieved_documents": documents,
                "runtime_seconds": round(
                    time.perf_counter() - question_started, 3),
                "config": {
                    "config_id": config_id,
                    "generator": generator_name,
                    "top_k": top_k,
                    "max_new_tokens": max_new_tokens,
                    "retriever": retriever_name,
                },
            }

            json.dump(record, output_file, ensure_ascii=False)
            output_file.write("\n")
            output_file.flush()

            if position % 10 == 0:
                elapsed = time.perf_counter() - started
                rate = elapsed / position
                remaining = (len(todo) - position) * rate / 60
                print(f"  {position}/{len(todo)}  "
                      f"{rate:.1f}s/question  "
                      f"~{remaining:.0f} min left", flush=True)

    total = (time.perf_counter() - started) / 60
    print(f"\nCONFIG {config_id} finished in {total:.1f} min "
          f"({failures} failures)")
    print(f"  -> {output_path}")


if __name__ == "__main__":
    main()