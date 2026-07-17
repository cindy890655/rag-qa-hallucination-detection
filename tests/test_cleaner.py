from config import WIKI_DUMP_PATH
from rag.cleaner import clean_wikipedia_text
from rag.wiki_loader import load_wikipedia_articles


def main() -> None:
    articles = load_wikipedia_articles(
        dump_path=WIKI_DUMP_PATH,
        max_articles=1
    )

    article = next(articles)

    raw_content = article["content"]
    cleaned_content = clean_wikipedia_text(raw_content)

    print("=" * 70)
    print("TITLE")
    print(article["title"])

    print("\n" + "=" * 70)
    print("BEFORE CLEANING")
    print(raw_content[:1500])

    print("\n" + "=" * 70)
    print("AFTER CLEANING")
    print(cleaned_content[:1500])

    print("\n" + "=" * 70)
    print(f"Raw length: {len(raw_content):,} characters")
    print(f"Cleaned length: {len(cleaned_content):,} characters")


if __name__ == "__main__":
    main()