import json
from time import perf_counter

import numpy as np
from sentence_transformers import SentenceTransformer

from config import (
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_MODEL,
    WIKI_OUTPUT_DIR,
)


def load_chunks_jsonl(input_path):
    """
    Load chunk records from a JSONL file.
    """
    chunks = []

    with input_path.open(mode="r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                chunk = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number}"
                ) from error

            if "title" not in chunk or "content" not in chunk:
                raise ValueError(
                    f"Chunk on line {line_number} must contain "
                    "'title' and 'content'"
                )

            chunks.append(chunk)

    return chunks


def main() -> None:
    chunks_path = WIKI_OUTPUT_DIR / "chunks.jsonl"
    embeddings_path = WIKI_OUTPUT_DIR / "embeddings.npy"

    if not chunks_path.exists():
        raise FileNotFoundError(
            f"Chunk file not found: {chunks_path}"
        )

    print("=" * 60)
    print("WIKIPEDIA EMBEDDING BUILD")
    print("=" * 60)
    print(f"Input: {chunks_path}")
    print(f"Model: {EMBEDDING_MODEL}")
    print(f"Batch size: {EMBEDDING_BATCH_SIZE}")

    print("\nLoading chunks...")
    load_start = perf_counter()

    chunks = load_chunks_jsonl(chunks_path)
    texts = [chunk["content"] for chunk in chunks]

    load_seconds = perf_counter() - load_start

    print(f"Loaded {len(chunks):,} chunks")
    print(f"Loading time: {load_seconds:.2f} seconds")

    print("\nLoading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print("\nGenerating embeddings...")
    embedding_start = perf_counter()

    embeddings = model.encode(
        texts,
        batch_size=EMBEDDING_BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    embedding_seconds = perf_counter() - embedding_start

    print(f"\nEmbedding shape: {embeddings.shape}")
    print(
        f"Embedding time: {embedding_seconds:.2f} seconds "
        f"({embedding_seconds / 60:.2f} minutes)"
    )

    print(f"\nSaving embeddings to:\n{embeddings_path}")
    save_start = perf_counter()

    np.save(embeddings_path, embeddings)

    save_seconds = perf_counter() - save_start
    total_seconds = load_seconds + embedding_seconds + save_seconds

    print(f"Save time: {save_seconds:.2f} seconds")
    print(
        f"Total measured time: {total_seconds:.2f} seconds "
        f"({total_seconds / 60:.2f} minutes)"
    )
    print("Embedding build completed successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()