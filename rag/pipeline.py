from rag.generator import LocalGenerator


class RAGPipeline:
    """
    Coordinate retrieval, prompt construction, and answer generation.
    """

    def __init__(
        self,
        retriever,
        generator: LocalGenerator
    ) -> None:
        self.retriever = retriever
        self.generator = generator

    def build_prompt(
        self,
        question: str,
        retrieved_chunks: list[dict]
    ) -> str:
        """
        Build a grounded QA prompt from retrieved context.
        """
        if not question.strip():
            raise ValueError("question cannot be empty")

        if not retrieved_chunks:
            raise ValueError("retrieved_chunks cannot be empty")

        context_sections = []

        for index, chunk in enumerate(retrieved_chunks, start=1):
            context_sections.append(
                f"[Document {index}]\n"
                f"Title: {chunk['title']}\n"
                f"Content: {chunk['content']}"
            )

        context = "\n\n".join(context_sections)

        return f"""Use only the context below to answer the question.

Extract the answer directly from the context.
Do not add opinions, criticism, explanations, or outside information.
If the answer is not explicitly supported by the context, reply exactly:
I don't know.

Context:
{context}

Question:
{question}

Give one short factual sentence:
"""

    def generate(
        self,
        question: str,
        top_k: int = 3
    ) -> dict:
        """
        Run the complete RAG pipeline.
        """
        if not question.strip():
            raise ValueError("question cannot be empty")

        if top_k <= 0:
            raise ValueError("top_k must be positive")

        retrieved_chunks = self.retriever.retrieve(
            question=question,
            top_k=top_k
        )

        prompt = self.build_prompt(
            question=question,
            retrieved_chunks=retrieved_chunks
        )

        answer = self.generator.generate(prompt)

        return {
            "question": question,
            "answer": answer,
            "retrieved_documents": [
                {
                    "title": chunk["title"],
                    "content": chunk["content"],
                    "score": chunk.get("score")
                }
                for chunk in retrieved_chunks
            ]
        }