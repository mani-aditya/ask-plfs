import re

import numpy as np
from rank_bm25 import BM25Okapi

from src import config, indexing
from src.utils import load_jsonl


def tokenize(text):
    """Lowercase words and numbers; keeps decimals like 41.7 as one token."""
    return re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", text.lower())


class Retriever:
    def __init__(self):
        self.chunks = load_jsonl(config.CHUNKS_PATH)
        self.index = indexing.load_index(config.FAISS_PATH)
        self.embedder = indexing.load_embedder()
        assert self.index.ntotal == len(self.chunks), "Index and chunks are out of sync - rebuild the index"
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in self.chunks])

    # ---- the three search modes: each returns [(chunk_row, score), ...] ----
    def dense_search(self, query, k):
        vector = self.embedder.encode(
            [config.QUERY_PREFIX + query], normalize_embeddings=True
        ).astype("float32")
        scores, ids = self.index.search(vector, k)
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i != -1]

    def bm25_search(self, query, k):
        scores = self.bm25.get_scores(tokenize(query))
        top_ids = np.argsort(scores)[::-1][:k]
        return [(int(i), float(scores[i])) for i in top_ids]

    def hybrid_search(self, query, k, candidates=20, rrf_k=60):
        dense = self.dense_search(query, candidates)
        sparse = self.bm25_search(query, candidates)
        fused = {}
        for results in (dense, sparse):
            for rank, (idx, _) in enumerate(results):
                fused[idx] = fused.get(idx, 0.0) + 1.0 / (rrf_k + rank + 1)
        ranked = sorted(fused.items(), key=lambda item: item[1], reverse=True)
        return ranked[:k]

    def _limit_per_page(self, hits, k, max_per_page=2):
       kept, per_page = [], {}
       for idx, score in hits:
           page = self.chunks[idx]["page"]
           if per_page.get(page, 0) >= max_per_page:
               continue
           per_page[page] = per_page.get(page, 0) + 1
           kept.append((idx, score))
           if len(kept) == k:
               break
       return kept

    # ---- the public method everything else calls ----
    def search(self, query, k=config.TOP_K, mode="hybrid", max_per_page=2):
       wide = k * 3          # fetch extra, then trim so one page can't fill every slot
       if mode == "dense":
           hits = self.dense_search(query, wide)
       elif mode == "bm25":
           hits = self.bm25_search(query, wide)
       elif mode == "hybrid":
           hits = self.hybrid_search(query, wide)
       else:
           raise ValueError("mode must be 'dense', 'bm25' or 'hybrid'")
       hits = self._limit_per_page(hits, k, max_per_page)
       return [{**self.chunks[idx], "score": score} for idx, score in hits]
