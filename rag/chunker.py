def chunk_documents(
    documents: list[dict[str, str]],
    chunk_size: int = 300,
    overlap: int = 50
) -> list[dict[str, str]]:
    """
    Split documents into smaller chunks, preferably ending at whitespace.

    Input:
        documents: list of documents with title and content
        chunk_size: max number of characters per chunk
        overlap: number of overlapping characters between chunks

    Output:
        list of chunks with title and content
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    chunks = []

    for doc in documents:
        title = doc["title"]
        text = doc["content"].strip()

        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))

            if end < len(text):
                space = text.rfind(" ", start, end)
                if space > start:
                    end = space

            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append({
                    "title": title,
                    "content": chunk_text
                })

            if end == len(text):
                break

            start = max(end - overlap, 0)

    return chunks