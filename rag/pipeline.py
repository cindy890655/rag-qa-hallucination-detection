from rag.generator import LocalGenerator
from rag.retriever import SemanticRetriever


class RAGPipeline:
    """
    Coordinate retrieval, prompt construction, and answer generation.

    Input:
        A natural-language question

    Output:
        A dictionary containing the generated answer and retrieved evidence
    """

    def __init__(
        self,
        retriever: SemanticRetriever,
        generator: LocalGenerator
    ) -> None:
        """
        Initialize the RAG pipeline.

        Input:
            retriever: retrieves relevant document chunks
            generator: generates an answer from the constructed prompt
        """
        self.retriever = retriever
        self.generator = generator

    def build_prompt(
        self,
        question: str,
        retrieved_chunks: list[dict]
    ) -> str:
        """
        Build an LLM prompt from the question and retrieved chunks.

        Input:
            question: user's natural-language question
            retrieved_chunks: ranked chunks returned by the retriever

        Output:
            Formatted prompt string
        """
        if not question.strip():
            raise ValueError("question cannot be empty")

        if not retrieved_chunks:
            raise ValueError("retrieved_chunks cannot be empty")

        context_sections = []

        for index, chunk in enumerate(retrieved_chunks, start=1):
            section = (
                f"[Document {index}]\n"
                f"Title: {chunk['title']}\n"
                f"Content: {chunk['content']}"
            )
            context_sections.append(section)

        context = "\n\n".join(context_sections)

        return f"""Answer the question using only the provided context.
If the answer is not supported by the context, reply with "I don't know."

Context:
{context}

Question:
{question}

Answer:
"""

    def generate(
        self,
        question: str,
        top_k: int = 3
    ) -> dict:
        """
        Run the complete RAG pipeline.

        Input:
            question: user's natural-language question
            top_k: number of relevant chunks to retrieve

        Output:
            Dictionary containing:
            - question
            - answer
            - retrieved_documents
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
                    "content": chunk["content"]
                }
                for chunk in retrieved_chunks
            ]
        }