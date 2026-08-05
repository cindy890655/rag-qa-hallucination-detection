from typing import Any

from rag.generator import LocalGenerator


class RAGPipeline:
    """
    Coordinate retrieval, prompt construction, and answer generation.
    """

    def __init__(
        self,
        retriever: Any,
        generator: LocalGenerator
    ) -> None:
        self.retriever = retriever
        self.generator = generator

    def build_prompt(
        self,
        question: str,
        retrieved_chunks: list[dict],
        max_input_tokens: int = 512
    ) -> str:
        """
        Build a grounded QA prompt that fits within the generator's
        maximum input length.

        Retrieved chunks are included in ranked order. Context is trimmed
        by tokenizer tokens rather than characters or words.
        """
        question = question.strip()

        if not question:
            raise ValueError("question cannot be empty")

        if not retrieved_chunks:
            raise ValueError("retrieved_chunks cannot be empty")

        if max_input_tokens <= 0:
            raise ValueError("max_input_tokens must be positive")

        prefix = (
            "Use only the context below to answer the question.\n\n"
            "Extract the answer directly from the context.\n"
            "Do not add opinions, criticism, explanations, "
            "or outside information.\n"
            "If the answer is not explicitly supported by the context, "
            'reply exactly: "I don\'t know."\n\n'
            f"Question:\n{question}\n\n"
            "Context:\n"
        )

        suffix = "\n\nAnswer:"

        tokenizer = self.generator.tokenizer

        prefix_tokens = tokenizer.encode(
            prefix,
            add_special_tokens=False
        )

        suffix_tokens = tokenizer.encode(
            suffix,
            add_special_tokens=False
        )

        reserved_tokens = len(prefix_tokens) + len(suffix_tokens)

        # Leave a small safety margin for special tokens.
        context_budget = max_input_tokens - reserved_tokens - 2

        if context_budget <= 0:
            raise ValueError(
                "question and instructions exceed the model input limit"
            )

        context_sections: list[str] = []
        remaining_budget = context_budget
        remaining_documents = len(retrieved_chunks)

        for index, chunk in enumerate(retrieved_chunks, start=1):
            title = str(chunk.get("title", "")).strip()
            content = str(chunk.get("content", "")).strip()

            if not title or not content:
                remaining_documents -= 1
                continue

            header = (
                f"[Document {index}]\n"
                f"Title: {title}\n"
                "Content: "
            )

            header_tokens = tokenizer.encode(
                header,
                add_special_tokens=False
            )

            separator = "\n\n" if context_sections else ""
            separator_tokens = tokenizer.encode(
                separator,
                add_special_tokens=False
            )

            required_tokens = len(header_tokens) + len(separator_tokens)

            if required_tokens >= remaining_budget:
                break

            # Divide the remaining budget among the remaining documents.
            document_budget = (
                remaining_budget // max(remaining_documents, 1)
            )

            content_budget = max(
                document_budget - required_tokens,
                0
            )

            if content_budget <= 0:
                break

            content_tokens = tokenizer.encode(
                content,
                add_special_tokens=False
            )

            trimmed_tokens = content_tokens[:content_budget]

            trimmed_content = tokenizer.decode(
                trimmed_tokens,
                skip_special_tokens=True
            ).strip()

            if trimmed_content:
                section = (
                    f"{header}{trimmed_content}"
                )

                context_sections.append(section)

                used_tokens = (
                    required_tokens + len(trimmed_tokens)
                )
                remaining_budget -= used_tokens

            remaining_documents -= 1

        if not context_sections:
            raise ValueError(
                "no retrieved context fits within the model input limit"
            )

        context = "\n\n".join(context_sections)
        prompt = prefix + context + suffix

        return prompt

    def generate(
        self,
        question: str,
        top_k: int,
        max_new_tokens: int
    ) -> dict:
        """
        Run the complete RAG pipeline.
        """
        question = question.strip()

        if not question:
            raise ValueError("question cannot be empty")

        if top_k <= 0:
            raise ValueError("top_k must be positive")

        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")

        retrieved_chunks = self.retriever.retrieve(
            question=question,
            top_k=top_k
        )

        if not retrieved_chunks:
            return {
                "question": question,
                "answer": "I don't know.",
                "retrieved_documents": []
            }

        prompt = self.build_prompt(
            question=question,
            retrieved_chunks=retrieved_chunks,
            max_input_tokens=512
        )

        answer = self.generator.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens
        )

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