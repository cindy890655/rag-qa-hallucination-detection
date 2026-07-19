import json
from typing import TextIO

from config import (
    CHUNKS_PATH,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EVALUATION_QUESTIONS_PATH,
    EMBEDDING_MODEL,
    GENERATOR_MODEL,
    INDEX_PATH,
    MAX_NEW_TOKENS,
    RAG_RESULTS_PATH,
    TOP_K
)

from rag.embedder import TextEmbedder
from rag.generator import LocalGenerator
from rag.pipeline import RAGPipeline
from rag.wiki_retriever import WikipediaRetriever


def load_questions(questions_path) -> list[dict]:
    """
    Load evaluation questions from a JSON file.
    """
    if not questions_path.exists():
        raise FileNotFoundError(
            f"Question file not found: {questions_path}"
        )

    with questions_path.open(
        mode="r",
        encoding="utf-8"
    ) as input_file:
        questions = json.load(input_file)

    if not isinstance(questions, list) or not questions:
        raise ValueError(
            "The evaluation question file must contain a non-empty list"
        )

    for item in questions:
        if not isinstance(item, dict):
            raise ValueError(
                "Each evaluation item must be a JSON object"
            )

        if not item.get("question", "").strip():
            raise ValueError(
                "Each evaluation item must contain a question"
            )

    return questions


def print_result(result: dict) -> None:
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
        print(f"Title: {document['title']}")

        score = document.get("score")
        if score is not None:
            print(f"Score: {score:.4f}")

        print(f"Content: {document['content']}")


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

def add_experiment_config(result: dict) -> dict:
    """
    Add the current experiment settings to one RAG result.
    """
    result["experiment_config"] = {
        "top_k": TOP_K,
        "max_new_tokens": MAX_NEW_TOKENS,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "embedding_model": EMBEDDING_MODEL,
        "generator_model": GENERATOR_MODEL
    }

    return result


def main() -> None:
    """
    Run the Wikipedia RAG pipeline on the configured demo questions.
    """
    evaluation_items = load_questions(
    EVALUATION_QUESTIONS_PATH
)
    print("=" * 70)
    print("WIKIPEDIA RAG")
    print("=" * 70)
    print(f"Embedding model: {EMBEDDING_MODEL}")
    print(f"Generator model: {GENERATOR_MODEL}")
    print(f"Top-K: {TOP_K}")
    print(f"Number of questions: {len(evaluation_items)}")

    

    embedder = TextEmbedder(
        model_name=EMBEDDING_MODEL
    )

    retriever = WikipediaRetriever(
        chunks_path=CHUNKS_PATH,
        index_path=INDEX_PATH,
        embedder=embedder
    )

    generator = LocalGenerator(
        model_name=GENERATOR_MODEL
    )

    pipeline = RAGPipeline(
        retriever=retriever,
        generator=generator
    )

    RAG_RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with RAG_RESULTS_PATH.open(
        mode="w",
        encoding="utf-8"
    ) as output_file:

        for item in evaluation_items:
            question = item["question"]

            result = pipeline.generate(
                question=question,
                top_k=TOP_K,
                max_new_tokens=MAX_NEW_TOKENS
            )

            result["question_id"] = item.get("id")
            result["category"] = item.get("category")

            print_result(result)
            save_result(result, output_file)

    print("\n" + "=" * 70)
    print("ALL QUESTIONS COMPLETED")
    print(f"Results saved to: {RAG_RESULTS_PATH}")
    print("=" * 70)
    


if __name__ == "__main__":
    main()