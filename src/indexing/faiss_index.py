"""
FAISS-based vector index for fast nearest-neighbour retrieval.

Supports:
  - IndexFlatIP   : exact inner-product search (brute force)
  - IndexIVFFlat  : approximate search with inverted file index

Usage
-----
from src.indexing.faiss_index import FAISSIndex
index = FAISSIndex(mode="exact")
index.build(item_ids, item_embeddings)
candidates = index.search(user_embedding, k=100)
"""

import time
import numpy as np
import faiss
from typing import Dict, List, Tuple, Optional


class FAISSIndex:
    def __init__(self, mode: str = "exact", n_list: int = 100):
        """
        Parameters
        ----------
        mode   : "exact" → IndexFlatIP  |  "approx" → IndexIVFFlat
        n_list : number of Voronoi cells for IVF index (used when mode="approx")
        """
        assert mode in ("exact", "approx"), "mode must be 'exact' or 'approx'"
        self.mode   = mode
        self.n_list = n_list
        self.index: Optional[faiss.Index] = None
        self.item_ids: List[int] = []
        self.dim: int = 0

    # ------------------------------------------------------------------
    def build(self, item_ids: List[int], embeddings: np.ndarray):
        """
        Parameters
        ----------
        item_ids   : list of video_ids (length = N)
        embeddings : float32 array of shape (N, dim)
        """
        embeddings = np.asarray(embeddings, dtype=np.float32)
        faiss.normalize_L2(embeddings)          # normalise for cosine via IP

        self.item_ids = list(item_ids)
        self.dim      = embeddings.shape[1]

        if self.mode == "exact":
            self.index = faiss.IndexFlatIP(self.dim)
        else:
            quantizer  = faiss.IndexFlatIP(self.dim)
            self.index = faiss.IndexIVFFlat(quantizer, self.dim,
                                             self.n_list, faiss.METRIC_INNER_PRODUCT)
            self.index.train(embeddings)
            self.index.nprobe = max(1, self.n_list // 10)  # search 10% of cells

        self.index.add(embeddings)
        print(f"FAISS [{self.mode}] index built: {self.index.ntotal:,} vectors, dim={self.dim}")

    # ------------------------------------------------------------------
    def search(self, query: np.ndarray, k: int = 100) -> List[int]:
        """
        Parameters
        ----------
        query : float32 array of shape (dim,) or (1, dim)

        Returns
        -------
        list of video_ids (length = k)
        """
        query = np.asarray(query, dtype=np.float32)
        if query.ndim == 1:
            query = query[np.newaxis, :]
        faiss.normalize_L2(query)
        _, indices = self.index.search(query, k)
        return [self.item_ids[i] for i in indices[0] if i >= 0]

    def search_batch(self, queries: np.ndarray, k: int = 100) -> List[List[int]]:
        """
        queries : float32 array of shape (n_users, dim)
        Returns list of lists of video_ids.
        """
        queries = np.asarray(queries, dtype=np.float32)
        faiss.normalize_L2(queries)
        _, indices = self.index.search(queries, k)
        return [
            [self.item_ids[i] for i in row if i >= 0]
            for row in indices
        ]

    # ------------------------------------------------------------------
    def benchmark(self, query_embeddings: np.ndarray, k: int = 100,
                  n_queries: int = 200) -> Dict[str, float]:
        """
        Measure mean latency per query (ms) vs brute-force baseline.

        Parameters
        ----------
        query_embeddings : float32 (N, dim) — a sample of user embeddings
        n_queries        : how many random queries to run

        Returns
        -------
        dict with latency_ms_mean, latency_ms_p95
        """
        rng = np.random.default_rng(42)
        idx = rng.choice(len(query_embeddings), size=min(n_queries, len(query_embeddings)),
                         replace=False)
        queries = query_embeddings[idx].astype(np.float32)

        latencies = []
        for q in queries:
            t0 = time.perf_counter()
            self.search(q, k=k)
            latencies.append((time.perf_counter() - t0) * 1000)   # ms

        result = {
            "mode":           self.mode,
            "n_queries":      len(latencies),
            "latency_ms_mean": float(np.mean(latencies)),
            "latency_ms_p50":  float(np.percentile(latencies, 50)),
            "latency_ms_p95":  float(np.percentile(latencies, 95)),
        }
        print(f"[FAISS {self.mode}] mean={result['latency_ms_mean']:.2f}ms  "
              f"p95={result['latency_ms_p95']:.2f}ms  (n={len(latencies)})")
        return result

    # ------------------------------------------------------------------
    def save(self, index_path: str, ids_path: str):
        import os, json
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        faiss.write_index(self.index, index_path)
        with open(ids_path, "w") as f:
            json.dump({"item_ids": [int(x) for x in self.item_ids], "dim": self.dim, "mode": self.mode}, f)
        print(f"FAISS index saved to {index_path}")

    @staticmethod
    def load(index_path: str, ids_path: str) -> "FAISSIndex":
        import json
        obj = FAISSIndex()
        obj.index = faiss.read_index(index_path)
        with open(ids_path) as f:
            meta = json.load(f)
        obj.item_ids = meta["item_ids"]
        obj.dim      = meta["dim"]
        obj.mode     = meta["mode"]
        return obj
