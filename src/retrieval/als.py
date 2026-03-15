"""
ALS (Alternating Least Squares) retrieval model using the `implicit` library.

Trains on big_matrix with confidence weighting:
    confidence = 1 + alpha * watch_ratio   (alpha=40 by default)

Usage
-----
from src.retrieval.als import ALSRetriever
model = ALSRetriever()
model.fit(train_df)
recs = model.recommend_batch(user_ids, n=100)
"""

import numpy as np
import implicit
from scipy.sparse import csr_matrix
from typing import Dict, List

from src.data.preprocessing import build_id_maps, build_sparse_matrix


ALPHA = 40          # confidence weight multiplier
FACTORS = 128       # embedding dimension
REGULARIZATION = 0.01
ITERATIONS = 30
RANDOM_STATE = 42


class ALSRetriever:
    def __init__(self,
                 factors: int = FACTORS,
                 regularization: float = REGULARIZATION,
                 iterations: int = ITERATIONS,
                 alpha: float = ALPHA):
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        self.alpha = alpha

        self.model = None
        self.user2idx: Dict[int, int] = {}
        self.idx2user: Dict[int, int] = {}
        self.item2idx: Dict[int, int] = {}
        self.idx2item: Dict[int, int] = {}
        self.user_item_matrix: csr_matrix = None

    # ------------------------------------------------------------------
    def fit(self, train_df):
        """
        train_df must have columns: user_id, video_id, watch_ratio.
        """
        self.user2idx, self.idx2user, self.item2idx, self.idx2item = \
            build_id_maps(train_df)

        # Build confidence matrix  C = 1 + alpha * watch_ratio
        confidence_mat = build_sparse_matrix(
            train_df, self.user2idx, self.item2idx, value_col="watch_ratio"
        )
        confidence_mat = confidence_mat.multiply(self.alpha)
        confidence_mat.data += 1.0          # C = alpha*r + 1
        self.user_item_matrix = confidence_mat.astype(np.float32)

        self.model = implicit.als.AlternatingLeastSquares(
            factors=self.factors,
            regularization=self.regularization,
            iterations=self.iterations,
            random_state=RANDOM_STATE,
            use_gpu=False,
        )
        # implicit 0.7.x expects user×item matrix
        self.model.fit(self.user_item_matrix)
        print(f"ALS trained: {self.factors} factors, {self.iterations} iterations")

    # ------------------------------------------------------------------
    def get_user_embeddings(self) -> np.ndarray:
        return self.model.user_factors   # shape (n_users, factors)

    def get_item_embeddings(self) -> np.ndarray:
        return self.model.item_factors   # shape (n_items, factors)

    # ------------------------------------------------------------------
    def recommend(self, user_id: int, n: int = 100,
                  filter_already_liked: bool = True) -> List[int]:
        """Return top-n recommended video_ids for a single user."""
        if user_id not in self.user2idx:
            return []
        uidx = self.user2idx[user_id]
        item_ids, _ = self.model.recommend(
            uidx,
            self.user_item_matrix[uidx],
            N=n,
            filter_already_liked_items=filter_already_liked,
        )
        return [self.idx2item[i] for i in item_ids]

    def recommend_batch(self, user_ids: List[int], n: int = 100,
                        filter_already_liked: bool = True) -> Dict[int, List[int]]:
        """Return top-n recommendations for a list of user_ids."""
        results = {}
        for uid in user_ids:
            results[uid] = self.recommend(uid, n=n,
                                          filter_already_liked=filter_already_liked)
        return results

    # ------------------------------------------------------------------
    def save(self, path: str):
        import pickle, os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"ALS model saved to {path}")

    @staticmethod
    def load(path: str) -> "ALSRetriever":
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)
