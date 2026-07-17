import bz2
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterator


def load_wikipedia_articles(
    dump_path: str | Path,
    max_articles: int | None = None
) -> Iterator[dict[str, str]]:
    """
    Stream articles from a compressed Wikipedia XML dump.

    Input:
        dump_path: path to the official Wikipedia .xml.bz2 dump
        max_articles: optional maximum number of main-namespace articles

    Output:
        Iterator of dictionaries containing:
        - title
        - content
    """
    dump_path = Path(dump_path)

    if not dump_path.exists():
        raise FileNotFoundError(f"Wikipedia dump not found: {dump_path}")

    if max_articles is not None and max_articles <= 0:
        raise ValueError("max_articles must be positive or None")

    article_count = 0

    with bz2.open(dump_path, mode="rb") as dump_file:
        for _, element in ET.iterparse(dump_file, events=("end",)):
            if not element.tag.endswith("page"):
                continue

            title_element = element.find("./{*}title")
            namespace_element = element.find("./{*}ns")
            text_element = element.find("./{*}revision/{*}text")
            redirect_element = element.find("./{*}redirect")

            title = (
                title_element.text.strip()
                if title_element is not None and title_element.text
                else ""
            )

            namespace = (
                namespace_element.text
                if namespace_element is not None
                else None
            )

            content = (
                text_element.text.strip()
                if text_element is not None and text_element.text
                else ""
            )

            # Keep only normal encyclopedia articles.
            if (
                namespace == "0"
                and redirect_element is None
                and title
                and content
            ):
                yield {
                    "title": title,
                    "content": content
                }

                article_count += 1

                if (
                    max_articles is not None
                    and article_count >= max_articles
                ):
                    break

            element.clear()