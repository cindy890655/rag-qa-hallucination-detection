"""
Build the large question bank used by the scaled-up RAG evaluation

Why questions are generated from mid-article chunks
--------------------------------------------------
A first attempt generated questions from article titles ("What is Antares?").
A 12-question pilot through the real RAG pipeline showed that approach is
useless for comparing configurations: retrieval hit the gold article 12/12 times
and every answer was a verbatim copy of the lead sentence, so the hallucination
rate would be 0% for every configuration and the comparison table would be all
zeros.

This script therefore samples chunks from the BODY of articles (never the lead
chunk) and asks an LLM to write a question about a specific detail inside that
chunk. Retrieval then has to locate one particular chunk among 390k instead of
matching a title, which is what makes top_k, the retriever and the generator
actually separate.

The gold chunk index is recorded for every question, which gives two free
validity signals later: retrieval hit rate, and whether the detector flags more
hallucinations on questions where retrieval missed the gold chunk.

Reads "rag part" read-only (chunks.jsonl). Imports nothing from it.

Usage
-----
    export GEMINI_API_KEY='...'
    python3 build_question_bank.py                     # 200 + 40, Gemini
    python3 build_question_bank.py --source local      # no API key needed
    python3 build_question_bank.py --n-answerable 50   # smaller trial run

Progress is saved every 10 questions, so an interrupted run resumes where it
stopped.
"""
import argparse
import json
import random
import re
import sys
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent
DETECTION_ROOT = HERE.parent
RAG_CHUNKS = (
    DETECTION_ROOT.parent / "rag part" / "data" / "wiki" / "chunks.jsonl"
)
OUTPUT_PATH = DETECTION_ROOT / "data" / "rag_eval" / "questions_large.json"

SEED = 42
MIN_CHUNK_CHARS = 400
GEMINI_MODEL = "gemini-flash-lite-latest"
SAVE_EVERY = 10

CATEGORIES = {
    "person", "place", "science", "technology",
    "arts", "organization", "history", "other",
}

# Titles that do not yield sensible questions.
BAD_TITLE_PREFIX = (
    "List of", "Lists of", "Index of", "Outline of", "Timeline of",
    "Glossary of", "Comparison of", "History of", "Geography of",
    "Politics of", "Economy of", "Transport in", "Demographics of",
    "Telecommunications in", "Foreign relations of", "Bibliography of",
)

# A question is rejected if it only makes sense while looking at the chunk.
CONTEXT_DEPENDENT = (
    "passage", "the text", "this text", "the article", "this article",
    "above", "below", "document", "excerpt", "mentioned here", "the context",
)

QUESTION_PROMPT = """You are building an evaluation set for a Wikipedia question-answering system.

Below is one passage from the Wikipedia article "{title}".

Write ONE factual question that:
- can be answered using only this passage
- names its subject explicitly, so the question still makes sense to someone who cannot see the passage
- asks about a specific detail such as a date, name, number, place, cause or result, NOT about the general topic
- never refers to "the passage", "the text", "the article" or "above"
- is a single sentence of fewer than 25 words

Also assign one category from exactly this list:
person, place, science, technology, arts, organization, history, other

Reply with JSON only, no explanation and no code fences:
{{"question": "...", "category": "..."}}

PASSAGE:
{content}"""


# ---------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------

def load_body_chunks(path: Path) -> list[dict]:
    """
    Return every chunk that is NOT the first chunk of its article.

    The lead chunk is excluded because questions written from it are answerable
    by copying the opening sentence, which makes all configurations look equal.
    """
    if not path.exists():
        raise FileNotFoundError(f"Corpus not found: {path}")

    seen_titles: set[str] = set()
    body: list[dict] = []

    with path.open(encoding="utf-8") as input_file:
        for index, line in enumerate(input_file):
            line = line.strip()

            if not line:
                continue

            record = json.loads(line)
            title = record["title"]
            content = record["content"]

            is_lead = title not in seen_titles
            seen_titles.add(title)

            if is_lead:
                continue

            if title.startswith(BAD_TITLE_PREFIX):
                continue

            if len(content) < MIN_CHUNK_CHARS:
                continue

            body.append({
                "title": title,
                "content": content,
                "gold_chunk_index": index,
            })

    return body


# ---------------------------------------------------------------------
# Question generation back ends
# ---------------------------------------------------------------------

def parse_reply(reply: str) -> dict | None:
    """
    Pull the JSON object out of a model reply and validate the question.
    """
    text = reply.strip()

    # tolerate ```json ... ``` even though the prompt forbids it
    fence = re.search(r"\{.*\}", text, re.S)
    if not fence:
        return None

    try:
        parsed = json.loads(fence.group(0))
    except json.JSONDecodeError:
        return None

    question = str(parsed.get("question", "")).strip()
    category = str(parsed.get("category", "")).strip().lower()

    if not question.endswith("?"):
        return None

    word_count = len(question.split())
    if word_count < 5 or word_count > 30:
        return None

    lowered = question.lower()
    if any(marker in lowered for marker in CONTEXT_DEPENDENT):
        return None

    return {
        "question": question,
        "category": category if category in CATEGORIES else "other",
    }


class GeminiBackend:
    """Question generation through the Gemini API."""

    name = "gemini"

    def __init__(self, api_key: str) -> None:
        from google import genai

        self.client = genai.Client(api_key=api_key)

    def ask(self, prompt: str) -> str:
        response = self.client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )
        return response.text or ""


class LocalBackend:
    """Question generation with the local Phi-3 model, no API key required."""

    name = "local-phi3"

    def __init__(self) -> None:
        sys.path.insert(0, str(HERE))
        from fast_generator import FastGenerator

        self.generator = FastGenerator("phi3")

    def ask(self, prompt: str) -> str:
        return self.generator.generate(prompt, max_new_tokens=110)


def make_question(backend, chunk: dict) -> dict | None:
    """
    Ask the backend for one question about `chunk`, retrying once.
    """
    prompt = QUESTION_PROMPT.format(
        title=chunk["title"],
        content=chunk["content"],
    )

    for attempt in range(2):
        try:
            parsed = parse_reply(backend.ask(prompt))
        except Exception as error:
            print(f"    api error ({error}); waiting 20s")
            time.sleep(20)
            continue

        if parsed:
            return parsed

    return None


# ---------------------------------------------------------------------
# Unanswerable questions
# ---------------------------------------------------------------------

# (subject, question). The subject is checked against the corpus titles and the
# pair is kept only if the corpus really cannot answer it.
ABSENT_TOPIC_QUESTIONS = [
    ("Mount Everest", "How tall is Mount Everest?"),
    ("Taj Mahal", "Who built the Taj Mahal?"),
    ("Machu Picchu", "Where is Machu Picchu located?"),
    ("Nelson Mandela", "Who was Nelson Mandela?"),
    ("Vincent van Gogh", "Who was Vincent van Gogh?"),
    ("Serena Williams", "Who is Serena Williams?"),
    ("Tokyo", "What is the population of Tokyo?"),
    ("Seoul", "In which country is Seoul located?"),
    ("Vietnam", "What is the capital of Vietnam?"),
    ("Peru", "Where is Peru located?"),
    ("Zimbabwe", "What is the capital of Zimbabwe?"),
    ("Uruguay", "Where is Uruguay located?"),
    ("Blockchain", "What is a blockchain?"),
    ("Quantum computing", "What is quantum computing?"),
    ("Quantum entanglement", "What is quantum entanglement?"),
    ("CRISPR", "What is CRISPR used for?"),
    ("Penicillin", "Who discovered penicillin?"),
    ("Photosynthesis", "What is photosynthesis?"),
    ("Machine learning", "What is machine learning?"),
    ("Neural network", "What is an artificial neural network?"),
    ("World Wide Web", "Who invented the World Wide Web?"),
    ("The Beatles", "Who were the members of the Beatles?"),
    ("Star Wars", "Who directed the first Star Wars film?"),
    ("Harry Potter", "Who wrote the Harry Potter books?"),
    ("Titanic", "When did the Titanic sink?"),
    ("Chernobyl disaster", "When did the Chernobyl disaster happen?"),
    ("Great Barrier Reef", "Where is the Great Barrier Reef?"),
    ("Yellowstone National Park", "Where is Yellowstone National Park?"),
    ("Sahara", "How large is the Sahara desert?"),
    ("Nile", "How long is the Nile river?"),
    ("Volleyball", "How many players are on a volleyball team?"),
    ("Table tennis", "How is a game of table tennis scored?"),
    ("Rugby", "How many players are on a rugby team?"),
    ("Sushi", "What is sushi made of?"),
    ("Pizza", "Where did pizza originate?"),
    ("Origami", "What is origami?"),
    ("Yoga", "Where did yoga originate?"),
    ("Zebra", "Where do zebras live?"),
    ("Penguin", "Where do penguins live?"),
    ("Octopus", "How many arms does an octopus have?"),
    ("Tulip", "Where do tulips grow?"),
    ("Bamboo", "How fast does bamboo grow?"),
    ("Vaccine", "How does a vaccine work?"),
    ("Solar panel", "How does a solar panel produce electricity?"),
]

# Unanswerable no matter how large the corpus is.
UNKNOWABLE_QUESTIONS = [
    "Who will win the 2034 FIFA World Cup?",
    "Who will be the first person to walk on Mars?",
    "What will the world population be in the year 2100?",
    "Which company will be the most valuable in 2050?",
    "What will be the next major scientific breakthrough?",
    "How many people will live in Tokyo in 2075?",
]


def build_unanswerable(corpus_titles: set[str], wanted: int) -> list[dict]:
    """
    Build questions the corpus cannot answer, used to measure abstention.
    """
    questions: list[dict] = []

    for subject, text in ABSENT_TOPIC_QUESTIONS:
        if subject in corpus_titles:
            # the corpus does have an article on it, so it is answerable
            continue

        questions.append({
            "question": text,
            "category": "unanswerable",
            "answerable": False,
            "reason": "absent-from-corpus",
            "gold_title": None,
            "gold_chunk_index": None,
        })

    for text in UNKNOWABLE_QUESTIONS:
        questions.append({
            "question": text,
            "category": "unanswerable",
            "answerable": False,
            "reason": "unknowable",
            "gold_title": None,
            "gold_chunk_index": None,
        })

    random.Random(SEED).shuffle(questions)
    return questions[:wanted]


# ---------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------

def load_progress(path: Path) -> list[dict]:
    if not path.exists():
        return []

    saved = json.load(path.open(encoding="utf-8"))
    done = [q for q in saved.get("questions", []) if q.get("answerable")]
    print(f"Resuming: {len(done)} answerable questions already generated")
    return done


def save(path: Path, answerable: list[dict], unanswerable: list[dict],
         source: str, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "description": (
            "Question bank for the scaled-up RAG evaluation. Answerable "
            "questions were generated by an LLM from non-lead chunks of the "
            "Wikipedia corpus, so each one targets a specific chunk rather "
            "than an article title. gold_chunk_index is the line number of "
            "that chunk in rag part/data/wiki/chunks.jsonl."
        ),
        "config": {
            "seed": seed,
            "question_source": source,
            "n_answerable": len(answerable),
            "n_unanswerable": len(unanswerable),
            "min_chunk_chars": MIN_CHUNK_CHARS,
        },
        "questions": answerable + unanswerable,
    }

    with path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["gemini", "local"],
                        default="gemini")
    parser.add_argument("--n-answerable", type=int, default=200)
    parser.add_argument("--n-unanswerable", type=int, default=40)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--sleep", type=float, default=5.0,
                        help="pause between API calls, Gemini only")
    args = parser.parse_args()

    print("Loading corpus (body chunks only)...")
    body = load_body_chunks(RAG_CHUNKS)
    corpus_titles = {chunk["title"] for chunk in body}
    print(f"  {len(body):,} candidate chunks from {len(corpus_titles):,} articles")

    unanswerable = build_unanswerable(corpus_titles, args.n_unanswerable)
    print(f"  {len(unanswerable)} unanswerable questions prepared")

    answerable = load_progress(OUTPUT_PATH)
    already_used = {q["gold_chunk_index"] for q in answerable}

    # Oversample: some chunks will be rejected, so draw more than needed.
    rng = random.Random(args.seed)
    pool = rng.sample(body, min(len(body), args.n_answerable * 3))
    pool = [c for c in pool if c["gold_chunk_index"] not in already_used]

    if args.source == "gemini":
        import os

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise SystemExit("Set GEMINI_API_KEY first, or use --source local")
        backend = GeminiBackend(api_key)
    else:
        backend = LocalBackend()

    print(f"\nGenerating with {backend.name} "
          f"(target {args.n_answerable} answerable)\n")

    rejected = 0

    for chunk in pool:
        if len(answerable) >= args.n_answerable:
            break

        result = make_question(backend, chunk)

        if not result:
            rejected += 1
            continue

        answerable.append({
            "question": result["question"],
            "category": result["category"],
            "answerable": True,
            "reason": None,
                        "gold_title": chunk["title"],
            "gold_chunk_index": chunk["gold_chunk_index"],
        })

        if len(answerable) % SAVE_EVERY == 0:
            save(OUTPUT_PATH, answerable, unanswerable,
                 backend.name, args.seed)
            print(f"  {len(answerable)}/{args.n_answerable} "
                  f"(rejected so far: {rejected})")

        if args.source == "gemini":
            time.sleep(args.sleep)

    # give every question a stable id
    for order, question in enumerate(answerable, start=1):
        question["id"] = f"ans_{order:04d}"

    for order, question in enumerate(unanswerable, start=1):
        question["id"] = f"una_{order:04d}"

    save(OUTPUT_PATH, answerable, unanswerable, backend.name, args.seed)

    print("\n" + "=" * 70)
    print("QUESTION BANK BUILT")
    print("=" * 70)
    print(f"answerable   : {len(answerable)}")
    print(f"unanswerable : {len(unanswerable)}")
    print(f"rejected     : {rejected}")
    print(f"output       : {OUTPUT_PATH}")
    print("\nFirst 8 answerable questions:")
    for question in answerable[:8]:
        print(f"  [{question['category']:<12}] {question['question']}")
        print(f"                 gold: {question['gold_title']}")


if __name__ == "__main__":
    main()