"""
Item2Vec retrieval model using Gensim's Word2Vec skip-gram.

Each user's positively-watched videos (watch_ratio >= threshold), sorted by
timestamp, form a "sentence". Word2Vec learns item embeddings from co-occurrence.

User embeddings are built by averaging the item embeddings of all positively-
watched items.

Usage
-----
from src.retrieval.item2vec import Item2VecRetriever
model = Item2VecRetriever()
model.fit(train_df)
recs = model.recommend_batch(user_ids, n=100)
"""

import numpy as np
from gensim.models import Word2Vec
from typing import Dict, List, Optional

from src.data.preprocessing import build_watch_sequences, ITEM2VEC_THRESHOLD


VECTOR_SIZE  = 128
WINDOW       = 1000     # large window = global co-occurrence
NEGATIVE     = 5
MIN_COUNT    = 2
WORKERS      = 4
EPOCHS       = 10
SEED         = 42


class Item2VecRetriever:
    def __init__(self,
                 vector_size: int = VECTOR_SIZE,
                 window: int = WINDOW,
                 negative: int = NEGATIVE,
                 min_count: int = MIN_COUNT,
                 workers: int = WORKERS,
                 epochs: int = EPOCHS,
                 threshold: float = ITEM2VEC_THRESHOLD):
        self.vector_size = vector_size
        self.window      = window
        self.negative    = negative
        self.min_count   = min_count
        self.workers     = workers
        self.epochs      = epochs
        self.threshold   = threshold

        self.w2v_model: Optional[Word2Vec] = None
        self.item_embeddings: Dict[int, np.ndarray] = {}
        self.user_embeddings: Dict[int, np.ndarray] = {}
        self.all_item_ids: List[int] = []
        self.all_item_matrix: Optional[np.ndarray] = None  # (n_items, dim)

    # ------------------------------------------------------------------
    def fit(self, train_df):
        """
        train_df must have columns: user_id, video_id, watch_ratio, timestamp.
        """
        sequences = build_watch_sequences(train_df, threshold=self.threshold)
        # Gensim expects list of list of strings
        sentences = [[str(vid) for vid in seq] for seq in sequences.values()]

        print(f"Training Word2Vec on {len(sentences):,} sequences "
              f"(vector_size={self.vector_size}, window={self.window})...")
        self.w2v_model = Word2Vec(
            sentences=sentences,
            vector_size=self.vector_size,
            window=self.window,
            sg=1,           # skip-gram
            negative=self.negative,
            min_count=self.min_count,
            workers=self.workers,
            epochs=self.epochs,
            seed=SEED,
        )
        # Build item embedding lookup
        self.item_embeddings = {
            int(word): self.w2v_model.wv[word]
            for word in self.w2v_model.wv.index_to_key
        }
        self.all_item_ids = list(self.item_embeddings.keys())
        self.all_item_matrix = np.stack(
            [self.item_embeddings[v] for v in self.all_item_ids]
        ).astype(np.float32)
        print(f"Item2Vec trained: {len(self.item_embeddings):,} item embeddings")

        # Build user embeddings from sequences
        self._build_user_embeddings(sequences)

    # ------------------------------------------------------------------
    def _build_user_embeddings(self, sequences: Dict[int, List[int]]):
        """Average item embeddings of positively-watched items per user."""
        self.user_embeddings = {}
        for user_id, seq in sequences.items():
            vecs = [self.item_embeddings[v] for v in seq if v in self.item_embeddings]
            if vecs:
                self.user_embeddings[user_id] = np.mean(vecs, axis=0).astype(np.float32)

        print(f"User embeddings built for {len(self.user_embeddings):,} users")

    # ------------------------------------------------------------------
    def recommend(self, user_id: int, n: int = 100) -> List[int]:
        """Return top-n recommended video_ids by cosine similarity."""
        if user_id not in self.user_embeddings:
            return []
        user_vec = self.user_embeddings[user_id]
        # Cosine similarity against all items
        norms = np.linalg.norm(self.all_item_matrix, axis=1) + 1e-9
        user_norm = np.linalg.norm(user_vec) + 1e-9
        sims = (self.all_item_matrix @ user_vec) / (norms * user_norm)
        top_idx = np.argsort(-sims)[:n]
        return [self.all_item_ids[i] for i in top_idx]

    def recommend_batch(self, user_ids: List[int], n: int = 100) -> Dict[int, List[int]]:
        return {uid: self.recommend(uid, n=n) for uid in user_ids}

    # ------------------------------------------------------------------
    def get_item_embeddings(self) -> tuple[List[int], np.ndarray]:
        """Return (item_ids list, embedding matrix) for FAISS indexing."""
        return self.all_item_ids, self.all_item_matrix

    def get_user_embedding(self, user_id: int) -> Optional[np.ndarray]:
        return self.user_embeddings.get(user_id)

    # ------------------------------------------------------------------
    def save(self, path: str):
        import pickle, os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"Item2Vec model saved to {path}")

    @staticmethod
    def load(path: str) -> "Item2VecRetriever":
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)
