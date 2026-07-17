import json
from pathlib import Path

import faiss
import numpy as np

from rag.embedder import TextEmbedder


class WikipediaRetriever:
    """
    Retrieve Wikipedia chunks from a saved FAISS index.
    """

    def __init__(
        self,
        chunks_path: str | Path,
        index_path: str | Path,
        embedder: TextEmbedder
    ) -> None:
        self.chunks_path = Path(chunks_path)
        self.index_path = Path(index_path)
        self.embedder = embedder

        if not self.chunks_path.exists():
            raise FileNotFoundError(
                f"Chunk file not found: {self.chunks_path}"
            )

        if not self.index_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found: {self.index_path}"
            )

        print("Loading Wikipedia chunks...")
        self.chunks = self._load_chunks(self.chunks_path)

        print("Loading FAISS index...")
        self.index = faiss.read_index(str(self.index_path))

        if len(self.chunks) != self.index.ntotal:
            raise ValueError(
                "Chunk count and FAISS vector count do not match: "
                f"{len(self.chunks):,} chunks vs. "
                f"{self.index.ntotal:,} vectors"
            )

        print(
            f"Wikipedia retriever loaded: "
            f"{len(self.chunks):,} chunks"
        )

    @staticmethod
    def _load_chunks(
        chunks_path: Path
    ) -> list[dict[str, str]]:
        """
        Load chunk metadata from a JSONL file.
        """
        chunks: list[dict[str, str]] = []

        with chunks_path.open(
            mode="r",
            encoding="utf-8"
        ) as input_file:
            for line_number, line in enumerate(
                input_file,
                start=1
            ):
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

    def retrieve(
        self,
        question: str,
        top_k: int = 3
    ) -> list[dict]:
        """
        Retrieve the top-k Wikipedia chunks for a question.
        """
        if not question.strip():
            raise ValueError("question cannot be empty")

        if top_k <= 0:
            raise ValueError("top_k must be positive")

        query_embedding = self.embedder.embed_query(question)

        query_embedding = np.ascontiguousarray(
            query_embedding.reshape(1, -1).astype("float32")
        )

        scores, indices = self.index.search(
            query_embedding,
            min(top_k, self.index.ntotal)
        )

        results = []

        for score, index in zip(scores[0], indices[0]):
            if index < 0:
                continue

            chunk = self.chunks[index]

            results.append({
                "title": chunk["title"],
                "content": chunk["content"],
                "score": float(score)
            })

        return results