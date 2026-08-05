from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------
# Wikipedia data
# ---------------------------------------------------------------------

# Local path to the English Wikipedia dump.
# Each team member should change this path to their own dump location.
WIKI_DUMP_PATH = Path(
    r"F:\Chrome Download\enwiki-20260701-pages-articles-multistream.xml.bz2"
)

# Directory for generated Wikipedia data files.
WIKI_OUTPUT_DIR = PROJECT_ROOT / "data" / "wiki"

# Maximum number of valid Wikipedia articles to process.
MAX_ARTICLES = 10000


# ---------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------

# Number of words in each chunk.
CHUNK_SIZE = 100

# Number of overlapping words between consecutive chunks.
CHUNK_OVERLAP = 20


# ---------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_BATCH_SIZE = 32


# ---------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------

# Number of chunks retrieved for each question.
TOP_K = 3


# ---------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------

AVAILABLE_GENERATORS = {
    "flan-base": "google/flan-t5-base",
    "flan-large": "google/flan-t5-large",
    "phi3": "microsoft/Phi-3-mini-4k-instruct",
}

# Select the generator model for the experiment.
GENERATOR_NAME = "phi3"

GENERATOR_MODEL = AVAILABLE_GENERATORS[GENERATOR_NAME]

MAX_NEW_TOKENS = 64


# ---------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------

CHUNKS_PATH = WIKI_OUTPUT_DIR / "chunks.jsonl"
EMBEDDINGS_PATH = WIKI_OUTPUT_DIR / "embeddings.npy"
INDEX_PATH = WIKI_OUTPUT_DIR / "index.faiss"
RAG_RESULTS_PATH = WIKI_OUTPUT_DIR / (
    f"rag_results_k{TOP_K}_tokens{MAX_NEW_TOKENS}.jsonl"
)

EVALUATION_QUESTIONS_PATH = (
    PROJECT_ROOT / "data" / "evaluation" / "questions.json"
)