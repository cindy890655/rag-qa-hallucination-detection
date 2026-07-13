from sentence_transformers import SentenceTransformer


class TextEmbedder:
    """
    Convert text chunks and user questions into dense vector embeddings.

    Input:
        model_name: Hugging Face sentence-transformer model name

    Output:
        NumPy arrays containing normalized text embeddings
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model = SentenceTransformer(model_name)

    def embed_chunks(
        self,
        chunks: list[dict[str, str]]
    ) -> list[dict]:
        """
        Add an embedding vector to every text chunk.

        Input:
            chunks: list of dictionaries containing title and content

        Output:
            list of dictionaries containing title, content, and embedding
        """
        if not chunks:
            return []

        texts = [chunk["content"] for chunk in chunks]

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False
        )

        embedded_chunks = []

        for chunk, embedding in zip(chunks, embeddings):
            embedded_chunks.append({
                "title": chunk["title"],
                "content": chunk["content"],
                "embedding": embedding
            })

        return embedded_chunks

    def embed_query(self, question: str):
        """
        Convert one user question into a normalized embedding vector.

        Input:
            question: user question as a string

        Output:
            one NumPy embedding vector
        """
        if not question.strip():
            raise ValueError("question cannot be empty")

        return self.model.encode(
            question,
            normalize_embeddings=True,
            show_progress_bar=False
        )