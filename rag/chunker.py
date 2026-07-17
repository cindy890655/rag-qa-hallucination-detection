def chunk_documents(
    documents: list[dict[str, str]],
    chunk_size: int = 100,
    overlap: int = 20
) -> list[dict[str, str]]:
    """
    Split documents into word-based chunks.

    Input:
        documents: list of documents with title and content
        chunk_size: maximum number of words per chunk
        overlap: number of overlapping words between adjacent chunks

    Output:
        list of chunks with title and content
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be non-negative and smaller than chunk_size"
        )

    chunks: list[dict[str, str]] = []

    for document in documents:
        if "title" not in document or "content" not in document:
            raise ValueError(
                "each document must contain 'title' and 'content'"
            )

        title = document["title"].strip()
        text = document["content"].strip()

        if not title or not text:
            continue

        words = text.split()
        step = chunk_size - overlap

        for start in range(0, len(words), step):
            end = min(start + chunk_size, len(words))
            chunk_words = words[start:end]

            if not chunk_words:
                continue

            chunks.append({
                "title": title,
                "content": " ".join(chunk_words)
            })

            if end == len(words):
                break

    return chunks