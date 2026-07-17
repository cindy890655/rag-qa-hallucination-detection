from config import WIKI_OUTPUT_DIR
from rag.embedder import TextEmbedder
from rag.wiki_retriever import WikipediaRetriever


def main() -> None:
    embedder = TextEmbedder()

    retriever = WikipediaRetriever(
        chunks_path=WIKI_OUTPUT_DIR / "chunks.jsonl",
        index_path=WIKI_OUTPUT_DIR / "index.faiss",
        embedder=embedder
    )

    question = "What is anarchism?"

    results = retriever.retrieve(
        question=question,
        top_k=3
    )

    print(f"\nQuestion: {question}")
    print("\nTop retrieved chunks:")

    for rank, result in enumerate(results, start=1):
        print("\n" + "=" * 70)
        print(f"Rank {rank}")
        print(f"Title: {result['title']}")
        print(f"Score: {result['score']:.4f}")
        print(result["content"])


if __name__ == "__main__":
    main()