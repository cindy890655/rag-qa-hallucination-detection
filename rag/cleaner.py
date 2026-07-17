import html
import re
from collections.abc import Iterable


def _remove_nested_templates(text: str) -> str:
    """
    Remove MediaWiki templates enclosed by {{ ... }}.

    A small stack-based parser is used because templates may be nested,
    which ordinary regular expressions do not handle reliably.
    """
    output: list[str] = []
    template_depth = 0
    index = 0

    while index < len(text):
        if text.startswith("{{", index):
            template_depth += 1
            index += 2
            continue

        if text.startswith("}}", index) and template_depth > 0:
            template_depth -= 1
            index += 2
            continue

        if template_depth == 0:
            output.append(text[index])

        index += 1

    return "".join(output)


def clean_wikipedia_text(text: str) -> str:
    """
    Convert raw Wikipedia markup into cleaner natural-language text.

    Current cleaning rules:
        - Remove comments
        - Remove nested templates
        - Remove references and HTML tags
        - Remove file/image and category links
        - Keep the readable text from normal internal links
        - Remove external-link URLs while preserving optional labels
        - Simplify section headings and formatting marks
        - Normalize whitespace
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    if not text.strip():
        return ""

    cleaned = html.unescape(text)

    # Remove HTML comments.
    cleaned = re.sub(
        r"<!--.*?-->",
        " ",
        cleaned,
        flags=re.DOTALL
    )

    # Remove MediaWiki templates, including nested templates.
    cleaned = _remove_nested_templates(cleaned)

    # Remove complete reference blocks and standalone reference tags.
    cleaned = re.sub(
        r"<ref\b[^>]*>.*?</ref\s*>",
        " ",
        cleaned,
        flags=re.IGNORECASE | re.DOTALL
    )
    cleaned = re.sub(
        r"<ref\b[^>]*/\s*>",
        " ",
        cleaned,
        flags=re.IGNORECASE
    )

    # Remove tables written with {| ... |}.
    cleaned = re.sub(
        r"\{\|.*?\|\}",
        " ",
        cleaned,
        flags=re.DOTALL
    )

    # Remove file/image links entirely.
    cleaned = re.sub(
        r"\[\[(?:File|Image):.*?\]\]",
        " ",
        cleaned,
        flags=re.IGNORECASE | re.DOTALL
    )

    # Remove category links entirely.
    cleaned = re.sub(
        r"\[\[Category:.*?\]\]",
        " ",
        cleaned,
        flags=re.IGNORECASE | re.DOTALL
    )

    # Convert ordinary internal links:
    # [[United States]] -> United States
    # [[Python (programming language)|Python]] -> Python
    cleaned = re.sub(
        r"\[\[([^|\]]+)\|([^\]]+)\]\]",
        r"\2",
        cleaned
    )
    cleaned = re.sub(
        r"\[\[([^\]]+)\]\]",
        r"\1",
        cleaned
    )

    # External links:
    # [https://example.com visible label] -> visible label
    # [https://example.com] -> removed
    cleaned = re.sub(
        r"\[(?:https?://|ftp://)\S+\s+([^\]]+)\]",
        r"\1",
        cleaned
    )
    cleaned = re.sub(
        r"\[(?:https?://|ftp://)[^\]]+\]",
        " ",
        cleaned
    )

    # Remove remaining HTML/XML tags.
    cleaned = re.sub(
        r"<[^>]+>",
        " ",
        cleaned
    )

    # Convert headings such as == History == to History.
    cleaned = re.sub(
        r"^\s*=+\s*(.*?)\s*=+\s*$",
        r"\1",
        cleaned,
        flags=re.MULTILINE
    )

    # Remove bold and italic apostrophe markup.
    cleaned = cleaned.replace("'''", "")
    cleaned = cleaned.replace("''", "")

    # Remove common list markers at the beginning of lines.
    cleaned = re.sub(
        r"^\s*[*#:;]+\s*",
        "",
        cleaned,
        flags=re.MULTILINE
    )

    # Normalize spaces without destroying paragraph boundaries.
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r" *\n *", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    return cleaned.strip()


def clean_documents(
    documents: Iterable[dict[str, str]]
) -> list[dict[str, str]]:
    """
    Clean documents while preserving the title/content interface.

    Input:
        Iterable of dictionaries containing:
        - title
        - content

    Output:
        Cleaned documents containing the same two fields.
        Documents with empty titles or empty cleaned content are skipped.
    """
    cleaned_documents: list[dict[str, str]] = []

    for document in documents:
        if "title" not in document or "content" not in document:
            raise ValueError(
                "each document must contain 'title' and 'content'"
            )

        title = document["title"].strip()
        content = clean_wikipedia_text(document["content"])

        if title and content:
            cleaned_documents.append({
                "title": title,
                "content": content
            })

    return cleaned_documents