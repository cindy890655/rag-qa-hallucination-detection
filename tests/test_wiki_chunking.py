from config import (
    WIKI_DUMP_PATH,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)
from rag.chunker import chunk_documents
from rag.cleaner import clean_documents
from rag.wiki_loader import load_wikipedia_articles


def main() -> None:
    print("Loading Wikipedia...")

    documents = list(
        load_wikipedia_articles(
            dump_path=WIKI_DUMP_PATH,
            max_articles=3
        )
    )

    print(f"Loaded {len(documents)} articles")

    cleaned_documents = clean_documents(documents)
    print("Cleaning complete.")

    chunks = chunk_documents(
        cleaned_documents,
        chunk_size=CHUNK_SIZE,
        overlap=CHUNK_OVERLAP
    )

    print(f"Generated {len(chunks)} chunks\n")

    # Print the first three complete chunks.
    for index, chunk in enumerate(chunks[:3], start=1):
        words = chunk["content"].split()

        print("=" * 70)
        print(f"Chunk {index}")
        print(f"Title: {chunk['title']}")
        print(f"Word count: {len(words)}")
        print(chunk["content"])
        print()

    # Verify overlap between adjacent chunks from the same article.
    print("\n" + "=" * 70)
    print("OVERLAP CHECK")

    checked_pairs = 0

    for index in range(len(chunks) - 1):
        current_chunk = chunks[index]
        next_chunk = chunks[index + 1]

        # Do not compare chunks belonging to different articles.
        if current_chunk["title"] != next_chunk["title"]:
            continue

        current_words = current_chunk["content"].split()
        next_words = next_chunk["content"].split()

        current_overlap = current_words[-CHUNK_OVERLAP:]
        next_overlap = next_words[:CHUNK_OVERLAP]

        print("\n" + "-" * 70)
        print(f"Chunk {index + 1} ending:")
        print(" ".join(current_overlap))

        print(f"\nChunk {index + 2} beginning:")
        print(" ".join(next_overlap))

        print(
            "\nOverlap matches:",
            current_overlap == next_overlap
        )

        checked_pairs += 1

        if checked_pairs >= 3:
            break

        print("\n" + "=" * 70)
    print("RECONSTRUCTION CHECK")

    first_title = chunks[0]["title"]

    article_chunks = [
        chunk for chunk in chunks
        if chunk["title"] == first_title
    ]

    reconstructed_words: list[str] = []

    for index, chunk in enumerate(article_chunks):
        words = chunk["content"].split()

        if index == 0:
            reconstructed_words.extend(words)
        else:
            reconstructed_words.extend(
                words[CHUNK_OVERLAP:]
            )

    original_document = next(
        document
        for document in cleaned_documents
        if document["title"] == first_title
    )

    original_words = original_document["content"].split()

    print(f"Original word count: {len(original_words)}")
    print(f"Reconstructed word count: {len(reconstructed_words)}")
    print(
        "Full reconstruction matches:",
        original_words == reconstructed_words
    )


if __name__ == "__main__":
    main()