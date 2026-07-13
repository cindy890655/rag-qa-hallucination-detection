import numpy as np

from rag.embedder import TextEmbedder


class SemanticRetriever:
    """
    Retrieve the document chunks that are most semantically similar
    to a user's question.

    Input:
        embedded_chunks: chunks containing title, content, and embedding
        embedder: TextEmbedder used to embed the user question

    Output:
        A ranked list of the most relevant chunks with similarity scores
    """

    def __init__(
        self,
        embedded_chunks: list[dict],
        embedder: TextEmbedder
    ) -> None:
        if not embedded_chunks:
            raise ValueError("embedded_chunks cannot be empty")

        self.embedded_chunks = embedded_chunks
        self.embedder = embedder

    def retrieve(
        self,
        question: str,
        top_k: int = 3
    ) -> list[dict]:
        """
        Retrieve the top-k chunks most relevant to a question.

        Input:
            question: user's natural-language question
            top_k: maximum number of chunks to return

        Output:
            Ranked chunks containing:
            - title
            - content
            - score
        """
        if not question.strip():
            raise ValueError("question cannot be empty")

        if top_k <= 0:
            raise ValueError("top_k must be positive")

        query_embedding = self.embedder.embed_query(question)

        results = []

        for chunk in self.embedded_chunks:
            chunk_embedding = chunk["embedding"]

            # Both vectors were normalized in embedder.py.
            # Therefore, their dot product equals cosine similarity.
            similarity = float(
                np.dot(query_embedding, chunk_embedding)
            )

            results.append({
                "title": chunk["title"],
                "content": chunk["content"],
                "score": similarity
            })

        results.sort(
            key=lambda result: result["score"],
            reverse=True
        )

        return results[:min(top_k, len(results))]