from pathlib import Path


def load_text_documents(folder_path: str) -> list[dict[str, str]]:
    """
    Load all .txt files from a folder.

    Input:
        folder_path: path to a folder containing .txt documents

    Output:
        A list of dictionaries. Each dictionary has:
        - title: file name without extension
        - content: text content of the file
    """
    folder = Path(folder_path)

    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    documents = []

    for file_path in folder.glob("*.txt"):
        text = file_path.read_text(encoding="utf-8").strip()

        if text:
            documents.append({
                "title": file_path.stem,
                "content": text
            })

    return documents