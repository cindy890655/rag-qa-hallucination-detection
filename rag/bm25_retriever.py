import json
import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi


class BM25Retriever:
    """
    Retrieve Wikipedia chunks using BM25 keyword matching.

    The retriever uses the same preprocessed Wikipedia chunks as the
    dense FAISS retriever so that only the retrieval method changes.
    """

    def __init__(
        self,
        chunks_path: str | Path
    ) -> None:
        self.chunks_path = Path(chunks_path)

        if not self.chunks_path.exists():
            raise FileNotFoundError(
                f"Chunk file not found: {self.chunks_path}"
            )

        print("Loading Wikipedia chunks for BM25...")
        self.chunks = self._load_chunks(
            self.chunks_path
        )

        if not self.chunks:
            raise ValueError(
                "No valid chunks were loaded"
            )

        print("Tokenizing Wikipedia chunks for BM25...")

        tokenized_corpus = [
            self._tokenize(chunk["content"])
            for chunk in self.chunks
        ]

        print("Building BM25 index...")

        self.index = BM25Okapi(
            tokenized_corpus
        )

        print(
            f"BM25 retriever loaded: "
            f"{len(self.chunks):,} chunks"
        )

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """
        Convert text into lowercase word tokens for BM25.

        The tokenizer keeps letters, numbers, and underscore characters
        and removes punctuation.
        """
        return re.findall(
            r"\b\w+\b",
            text.lower()
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

                if (
                    "title" not in chunk
                    or "content" not in chunk
                ):
                    raise ValueError(
                        f"Chunk on line {line_number} "
                        "must contain 'title' and 'content'"
                    )

                title = str(
                    chunk["title"]
                ).strip()

                content = str(
                    chunk["content"]
                ).strip()

                if not title or not content:
                    continue

                chunks.append({
                    "title": title,
                    "content": content,
                })

        return chunks

    def retrieve(
        self,
        question: str,
        top_k: int = 3
    ) -> list[dict]:
        """
        Retrieve the top-k Wikipedia chunks for a question.
        """
        question = question.strip()

        if not question:
            raise ValueError(
                "question cannot be empty"
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be positive"
            )

        query_tokens = self._tokenize(
            question
        )

        if not query_tokens:
            return []

        scores = np.asarray(
            self.index.get_scores(query_tokens),
            dtype=np.float32
        )

        if scores.size == 0:
            return []

        result_count = min(
            top_k,
            len(self.chunks)
        )

        # Select the highest-scoring chunks efficiently.
        if result_count == len(self.chunks):
            top_indices = np.argsort(scores)[::-1]
        else:
            candidate_indices = np.argpartition(
                scores,
                -result_count
            )[-result_count:]

            top_indices = candidate_indices[
                np.argsort(
                    scores[candidate_indices]
                )[::-1]
            ]

        results: list[dict] = []

        for index in top_indices:
            chunk = self.chunks[int(index)]

            results.append({
                "title": chunk["title"],
                "content": chunk["content"],
                "score": float(scores[index]),
            })

        return results