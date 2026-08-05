import json
import time
from pathlib import Path
from typing import TextIO

from config import (
    CHUNKS_PATH,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EVALUATION_QUESTIONS_PATH,
    GENERATOR_MODEL,
    MAX_NEW_TOKENS,
    PROJECT_ROOT,
    TOP_K,
)

from rag.bm25_retriever import BM25Retriever
from rag.generator import LocalGenerator
from rag.pipeline import RAGPipeline


BM25_RESULTS_PATH = (
    PROJECT_ROOT
    / "data"
    / "wiki"
    / f"rag_results_bm25_k{TOP_K}_tokens"
      f"{MAX_NEW_TOKENS}.jsonl"
)


def load_questions(
    questions_path: Path
) -> list[dict]:
    """
    Load evaluation questions from a JSON file.
    """
    if not questions_path.exists():
        raise FileNotFoundError(
            f"Question file not found: "
            f"{questions_path}"
        )

    with questions_path.open(
        mode="r",
        encoding="utf-8"
    ) as input_file:
        questions = json.load(
            input_file
        )

    if (
        not isinstance(questions, list)
        or not questions
    ):
        raise ValueError(
            "The evaluation question file "
            "must contain a non-empty list"
        )

    for item in questions:
        if not isinstance(item, dict):
            raise ValueError(
                "Each evaluation item "
                "must be a JSON object"
            )

        if not item.get(
            "question",
            ""
        ).strip():
            raise ValueError(
                "Each evaluation item "
                "must contain a question"
            )

    return questions


def print_result(
    result: dict
) -> None:
    """
    Print one RAG result in a readable format.
    """
    print("\n" + "=" * 70)
    print("QUESTION")
    print(result["question"])

    print("\nANSWER")
    print(result["answer"])

    print("\nRETRIEVED DOCUMENTS")

    for rank, document in enumerate(
        result["retrieved_documents"],
        start=1
    ):
        print("\n" + "-" * 70)
        print(f"Rank: {rank}")
        print(
            f"Title: {document['title']}"
        )

        score = document.get("score")

        if score is not None:
            print(f"Score: {score:.4f}")

        print(
            f"Content: {document['content']}"
        )


def save_result(
    result: dict,
    output_file: TextIO
) -> None:
    """
    Save one RAG result as one JSONL record.
    """
    json.dump(
        result,
        output_file,
        ensure_ascii=False
    )

    output_file.write("\n")


def add_experiment_config(
    result: dict
) -> dict:
    """
    Add the current BM25 experiment settings.
    """
    result["experiment_config"] = {
        "retrieval_method": "BM25",
        "top_k": TOP_K,
        "max_new_tokens": MAX_NEW_TOKENS,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "generator_model": GENERATOR_MODEL,
    }

    return result


def main() -> None:
    """
    Run the Wikipedia RAG pipeline with BM25 retrieval.
    """
    evaluation_items = load_questions(
        EVALUATION_QUESTIONS_PATH
    )

    print("=" * 70)
    print("WIKIPEDIA BM25 RAG EXPERIMENT")
    print("=" * 70)
    print("Retrieval method: BM25")
    print(
        f"Generator model: {GENERATOR_MODEL}"
    )
    print(f"Top-K: {TOP_K}")
    print(f"Chunk size: {CHUNK_SIZE}")
    print(
        f"Chunk overlap: {CHUNK_OVERLAP}"
    )
    print(
        f"Max new tokens: {MAX_NEW_TOKENS}"
    )
    print(
        "Number of questions: "
        f"{len(evaluation_items)}"
    )

    retriever = BM25Retriever(
        chunks_path=CHUNKS_PATH
    )

    generator = LocalGenerator(
        model_name=GENERATOR_MODEL
    )

    pipeline = RAGPipeline(
        retriever=retriever,
        generator=generator
    )

    BM25_RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    total_start_time = time.perf_counter()
    question_times: list[float] = []

    with BM25_RESULTS_PATH.open(
        mode="w",
        encoding="utf-8"
    ) as output_file:

        for item in evaluation_items:
            question = item["question"]

            question_start_time = (
                time.perf_counter()
            )

            result = pipeline.generate(
                question=question,
                top_k=TOP_K,
                max_new_tokens=MAX_NEW_TOKENS
            )

            elapsed_seconds = (
                time.perf_counter()
                - question_start_time
            )

            question_times.append(
                elapsed_seconds
            )

            result["question_id"] = item.get(
                "id"
            )

            result["category"] = item.get(
                "category"
            )

            result["runtime_seconds"] = round(
                elapsed_seconds,
                4
            )

            result = add_experiment_config(
                result
            )

            print_result(result)

            print(
                "\nRuntime: "
                f"{elapsed_seconds:.4f} seconds"
            )

            save_result(
                result=result,
                output_file=output_file
            )

    total_runtime = (
        time.perf_counter()
        - total_start_time
    )

    average_runtime = (
        sum(question_times)
        / len(question_times)
        if question_times
        else 0.0
    )

    print("\n" + "=" * 70)
    print("ALL QUESTIONS COMPLETED")
    print(
        "Number of questions: "
        f"{len(evaluation_items)}"
    )
    print(
        f"Total runtime: "
        f"{total_runtime:.4f} seconds"
    )
    print(
        "Average runtime per question: "
        f"{average_runtime:.4f} seconds"
    )
    print(
        f"Results saved to: "
        f"{BM25_RESULTS_PATH}"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()