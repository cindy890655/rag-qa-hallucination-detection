# Wikipedia RAG Pipeline
This project builds an end-to-end Retrieval-Augmented Generation (RAG) system using a Wikipedia dump, MiniLM embeddings, FAISS retrieval, and FLAN-T5 generation.

## Project Setup
Install the required Python packages:
pip install sentence-transformers transformers torch faiss-cpu numpy tqdm mwxml

Download the English Wikipedia articles dump from Wikimedia.
enwiki-20260701-pages-articles-multistream.xml.bz2


IMPORTANT:
Open config.py:
modify:
WIKI_DUMP_PATH = Path(
    r"F:\Chrome Download\enwiki-20260701-pages-articles-multistream.xml.bz2"
)
As your own wiki dump current location.

Also configure the number of articles:
MAX_ARTICLES = 10000
CHUNK_SIZE = 100
CHUNK_OVERLAP = 20
Modify as you need.

How to set up the whole database for model:
1. python preprocess.py
   this will produce data/wiki/chunks.jsonl
2. python build_embeddings.py
   produce data/wiki/embeddings.npy
   Note: Generating embeddings for 10,000 Wikipedia articles may take about 1 hour or more on a typical desktop computer.
3.python build_index.py
   produce data/wiki/index.faiss
4.python demo.py
   produce data/wiki/rag_results.jsonl

if you need to change the question go demo.py and find:
questions = [
        "What is anarchism?",
        "Where is Alabama located?",
        "What is albedo?",
        "What is an atom?",
        "What is an academy?"
    ]
to change it.



Optional if needed:
Running Tests
Run individual tests from the project root using:

python -m tests.test_cleaner
python -m tests.test_wiki_loader
python -m tests.test_wiki_chunking
python -m tests.test_embedding
python -m tests.test_wiki_retriever
python -m tests.test_generator
