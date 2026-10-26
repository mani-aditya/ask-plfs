import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src import config
from src.utils import load_jsonl


def load_embedder():
    return SentenceTransformer(config.EMBEDDING_MODEL)


def embed_texts(model, texts, batch_size=64):
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return np.asarray(vectors, dtype="float32")


def build_faiss_index(vectors):
    index = faiss.IndexFlatIP(vectors.shape[1])   # inner product on normalised vectors = cosine
    index.add(vectors)
    return index


def save_index(index, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(path))


def load_index(path):
    return faiss.read_index(str(path))


def build_and_save_index():
    chunks = load_jsonl(config.CHUNKS_PATH)
    model = load_embedder()
    vectors = embed_texts(model, [c["text"] for c in chunks])
    index = build_faiss_index(vectors)
    save_index(index, config.FAISS_PATH)
    print(f"Indexed {index.ntotal} chunks, dimension {vectors.shape[1]}")
    return index
