import json
from pathlib import Path
from time import perf_counter

import faiss
import numpy as np

from config import WIKI_OUTPUT_DIR


def count_jsonl_records(path: Path) -> int:
    """
    Count non-empty records in a JSONL file.
    """
    count = 0

    with path.open("r", encoding="utf-8") as input_file:
        for line in input_file:
            if line.strip():
                count += 1

    return count


def main() -> None:
    chunks_path = WIKI_OUTPUT_DIR / "chunks.jsonl"
    embeddings_path = WIKI_OUTPUT_DIR / "embeddings.npy"
    index_path = WIKI_OUTPUT_DIR / "index.faiss"

    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunk file not found: {chunks_path}")

    if not embeddings_path.exists():
        raise FileNotFoundError(
            f"Embedding file not found: {embeddings_path}"
        )

    print("=" * 60)
    print("FAISS INDEX BUILD")
    print("=" * 60)

    print("\nLoading embeddings...")
    start = perf_counter()

    embeddings = np.load(embeddings_path)

    if embeddings.ndim != 2:
        raise ValueError(
            "embeddings.npy must contain a 2-dimensional array"
        )

    embeddings = np.ascontiguousarray(
        embeddings.astype("float32")
    )

    load_seconds = perf_counter() - start

    chunk_count = count_jsonl_records(chunks_path)

    print(f"Embedding shape: {embeddings.shape}")
    print(f"Chunk records: {chunk_count:,}")
    print(f"Loading time: {load_seconds:.2f} seconds")

    if chunk_count != embeddings.shape[0]:
        raise ValueError(
            "Chunk count and embedding count do not match: "
            f"{chunk_count:,} chunks vs. "
            f"{embeddings.shape[0]:,} embeddings"
        )

    dimension = embeddings.shape[1]

    # Embeddings were normalized during encoding.
    # Inner product therefore equals cosine similarity.
    index = faiss.IndexFlatIP(dimension)

    print("\nAdding embeddings to FAISS index...")
    build_start = perf_counter()

    index.add(embeddings)

    build_seconds = perf_counter() - build_start

    print(f"Vectors indexed: {index.ntotal:,}")
    print(f"Index dimension: {dimension}")
    print(f"Build time: {build_seconds:.2f} seconds")

    print(f"\nSaving index to:\n{index_path}")
    faiss.write_index(index, str(index_path))

    print("\nFAISS index built successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()