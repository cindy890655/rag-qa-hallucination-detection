import json
from pathlib import Path

from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    MAX_ARTICLES,
    WIKI_DUMP_PATH,
    WIKI_OUTPUT_DIR,
)
from rag.chunker import chunk_documents
from rag.cleaner import clean_documents
from rag.wiki_loader import load_wikipedia_articles


def save_chunks_jsonl(
    chunks: list[dict[str, str]],
    output_path: Path
) -> None:
    """
    Save chunks in JSON Lines format.

    Each line contains one chunk:
        {"title": "...", "content": "..."}
    """
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with output_path.open(
        mode="w",
        encoding="utf-8"
    ) as output_file:
        for chunk in chunks:
            json.dump(
                chunk,
                output_file,
                ensure_ascii=False
            )
            output_file.write("\n")


def main() -> None:
    print("=" * 60)
    print("WIKIPEDIA PREPROCESSING")
    print("=" * 60)

    print(f"Dump: {WIKI_DUMP_PATH}")
    print(f"Maximum articles: {MAX_ARTICLES:,}")
    print(
        f"Chunk settings: "
        f"{CHUNK_SIZE} words, "
        f"{CHUNK_OVERLAP} overlapping words"
    )

    print("\nLoading Wikipedia articles...")

    raw_documents = list(
        load_wikipedia_articles(
            dump_path=WIKI_DUMP_PATH,
            max_articles=MAX_ARTICLES
        )
    )

    print(f"Loaded {len(raw_documents):,} articles")

    print("\nCleaning articles...")

    cleaned_documents = clean_documents(raw_documents)

    print(
        f"Retained {len(cleaned_documents):,} "
        "non-empty cleaned articles"
    )

    print("\nCreating chunks...")

    chunks = chunk_documents(
        documents=cleaned_documents,
        chunk_size=CHUNK_SIZE,
        overlap=CHUNK_OVERLAP
    )

    print(f"Generated {len(chunks):,} chunks")

    output_path = WIKI_OUTPUT_DIR / "chunks.jsonl"

    print(f"\nSaving chunks to:\n{output_path}")

    save_chunks_jsonl(
        chunks=chunks,
        output_path=output_path
    )

    print("\nPreprocessing completed successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()