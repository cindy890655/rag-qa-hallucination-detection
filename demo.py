import json

from config import (
    WIKI_OUTPUT_DIR,
    EMBEDDING_MODEL
)
from rag.embedder import TextEmbedder
from rag.generator import LocalGenerator
from rag.pipeline import RAGPipeline
from rag.wiki_retriever import WikipediaRetriever


OUTPUT_PATH = WIKI_OUTPUT_DIR / "rag_results.jsonl"


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
    output_file
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


def main() -> None:
    print("=" * 70)
    print("WIKIPEDIA RAG")
    print("=" * 70)

    embedder = TextEmbedder(
        model_name=EMBEDDING_MODEL
    )

    retriever = WikipediaRetriever(
        chunks_path=WIKI_OUTPUT_DIR / "chunks.jsonl",
        index_path=WIKI_OUTPUT_DIR / "index.faiss",
        embedder=embedder
    )

    generator = LocalGenerator()

    pipeline = RAGPipeline(
        retriever=retriever,
        generator=generator
    )

    questions = [
        "What is anarchism?",
        "Where is Alabama located?",
        "What is albedo?",
        "What is an atom?",
        "What is an academy?"
    ]

    with OUTPUT_PATH.open(
        mode="w",
        encoding="utf-8"
    ) as output_file:

        for question in questions:
            result = pipeline.generate(
                question=question,
                top_k=1
            )

            print_result(result)
            save_result(result, output_file)

    print("\n" + "=" * 70)
    print("ALL QUESTIONS COMPLETED")
    print(f"Results saved to: {OUTPUT_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()