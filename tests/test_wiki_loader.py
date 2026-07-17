from config import MAX_ARTICLES, WIKI_DUMP_PATH
from rag.wiki_loader import load_wikipedia_articles


def main() -> None:
    articles = load_wikipedia_articles(
        dump_path=WIKI_DUMP_PATH,
        max_articles=MAX_ARTICLES
    )

    for index, article in enumerate(articles, start=1):
        print(f"\nArticle {index}")
        print(f"Title: {article['title']}")
        print(f"Content: {article['content'][:300]}")

        if index >= 5:
            break


if __name__ == "__main__":
    main()