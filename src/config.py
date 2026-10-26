from pathlib import Path

# ---------- Paths ----------
ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
INDEX_DIR = ROOT / "index"
PAGES_PATH = DATA_PROCESSED / "pages.jsonl"
CHUNKS_PATH = DATA_PROCESSED / "chunks.jsonl"
FAISS_PATH = INDEX_DIR / "chunks.faiss"
EVAL_PATH = ROOT / "eval" / "eval_questions.csv"

# ---------- Corpus ----------
REPORTS = {
    "plfs_annual_report_2025.pdf": {
        "label": "PLFS Annual Report 2025",
        "url": "https://raw.githubusercontent.com/mani-aditya/file_dump/main/plfs2025report.pdf",
    },
}

# ---------- Parsing and chunking ----------
CHUNK_SIZE_TOKENS = 256      # the embedding model reads at most 512; this leaves headroom
CHUNK_OVERLAP_TOKENS = 40
MIN_PAGE_CHARS = 200          # skip near-empty pages
MIN_ENGLISH_RATIO = 0.8       # skip Hindi / garbled pages
USE_MARKDOWN = True

# ---------- Embeddings ----------
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# ---------- Retrieval and generation ----------
TOP_K = 8
LLM_MODEL = "gemini-3.1-flash-lite"
FALLBACK_MODEL = "gemini-3.8-flash"
MAX_ANSWER_TOKENS = 800
