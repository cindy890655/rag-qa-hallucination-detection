from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

# Official English Wikipedia dump.
# Each team member only needs to change this local path.
WIKI_DUMP_PATH = Path(
    r"F:\Chrome Download\enwiki-20260701-pages-articles-multistream.xml.bz2"
)

# Processed Wikipedia output directory.
WIKI_OUTPUT_DIR = PROJECT_ROOT / "data" / "wiki"

# Process the first 50,000 valid main-namespace, non-redirect articles.
MAX_ARTICLES = 10000

# Word-based chunking settings.
CHUNK_SIZE = 100
CHUNK_OVERLAP = 20

# Embedding settings for the next stage.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_BATCH_SIZE = 32