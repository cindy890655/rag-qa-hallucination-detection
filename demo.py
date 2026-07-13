from rag.chunker import chunk_documents
from rag.embedder import TextEmbedder
from rag.generator import LocalGenerator
from rag.loader import load_text_documents
from rag.pipeline import RAGPipeline
from rag.retriever import SemanticRetriever


def main() -> None:
    documents = load_text_documents("data/sample_docs")

    chunks = chunk_documents(
        documents,
        chunk_size=80,
        overlap=20
    )

    embedder = TextEmbedder()
    embedded_chunks = embedder.embed_chunks(chunks)

    retriever = SemanticRetriever(
        embedded_chunks=embedded_chunks,
        embedder=embedder
    )

    generator = LocalGenerator()

    rag_pipeline = RAGPipeline(
        retriever=retriever,
        generator=generator
    )

    result = rag_pipeline.generate(
        question="Who invented Python?",
        top_k=3
    )

    print("=" * 60)
    print("QUESTION")
    print(result["question"])

    print("\nANSWER")
    print(result["answer"])

    print("\nRETRIEVED DOCUMENTS")

    for document in result["retrieved_documents"]:
        print(f"\nTitle: {document['title']}")
        print(f"Content: {document['content']}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()