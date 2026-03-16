"""
BPR-MF (Bayesian Personalized Ranking with Matrix Factorization) ranking model.

Takes Top-100 retrieval candidates and rescores them using learned user/item
embeddings with pairwise BPR loss.

Reference: Rendle et al., "BPR: Bayesian Personalized Ranking from Implicit
Feedback", UAI 2009.

Usage
-----
from src.ranking.bpr import BPRRanker
ranker = BPRRanker()
ranker.fit(train_df, candidate_sets)
ranked = ranker.rerank(user_id, candidate_video_ids)
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Set
import time

from src.data.preprocessing import build_id_maps


EMBEDDING_DIM = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
BATCH_SIZE = 4096
EPOCHS = 80
PATIENCE = 10
SEED = 42


class BPRMFModel(nn.Module):
    def __init__(self, n_users: int, n_items: int,
                 embedding_dim: int = EMBEDDING_DIM):
        super().__init__()
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)

        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        u_emb = self.user_embedding(users)
        i_emb = self.item_embedding(items)
        u_bias = self.user_bias(users).squeeze(-1)
        i_bias = self.item_bias(items).squeeze(-1)
        scores = (u_emb * i_emb).sum(dim=-1) + u_bias + i_bias
        return scores

    def bpr_loss(self, users: torch.Tensor,
                 pos_items: torch.Tensor,
                 neg_items: torch.Tensor) -> torch.Tensor:
        pos_scores = self.forward(users, pos_items)
        neg_scores = self.forward(users, neg_items)
        loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10).mean()

        # L2 regularization
        reg = (self.user_embedding(users).norm(2).pow(2) +
               self.item_embedding(pos_items).norm(2).pow(2) +
               self.item_embedding(neg_items).norm(2).pow(2)) / len(users)
        return loss + 1e-5 * reg


class BPRRanker:
    def __init__(self,
                 embedding_dim: int = EMBEDDING_DIM,
                 lr: float = LEARNING_RATE,
                 weight_decay: float = WEIGHT_DECAY,
                 batch_size: int = BATCH_SIZE,
                 epochs: int = EPOCHS,
                 patience: int = PATIENCE,
                 device: str = "auto"):
        self.embedding_dim = embedding_dim
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model: Optional[BPRMFModel] = None
        self.user2idx: Dict[int, int] = {}
        self.idx2user: Dict[int, int] = {}
        self.item2idx: Dict[int, int] = {}
        self.idx2item: Dict[int, int] = {}

    def fit(self, train_df, candidate_sets: Dict[int, List[int]] = None,
            log_path: str = None):
        """
        Train BPR-MF ranker.

        Parameters
        ----------
        train_df : DataFrame with user_id, video_id, watch_ratio
        candidate_sets : optional dict {user_id: [candidate video_ids]}
            If provided, training only uses interactions within candidate sets
            (i.e., the Top-100 retrieval candidates per user).
        """
        torch.manual_seed(SEED)
        np.random.seed(SEED)

        log_lines = []
        def log(msg):
            print(msg)
            log_lines.append(msg)

        log(f"BPR-MF training on {self.device}")

        # Filter to candidate sets if provided
        if candidate_sets is not None:
            import pandas as pd
            pairs = [(uid, vid) for uid, items in candidate_sets.items() for vid in items]
            valid_df = pd.DataFrame(pairs, columns=["user_id", "video_id"]).drop_duplicates()
            filtered_df = train_df.merge(valid_df, on=["user_id", "video_id"], how="inner")
            log(f"  Filtered to candidate sets: {len(filtered_df):,} / {len(train_df):,} interactions")
        else:
            filtered_df = train_df

        # Build ID maps from filtered data
        self.user2idx, self.idx2user, self.item2idx, self.idx2item = \
            build_id_maps(filtered_df)
        n_users = len(self.user2idx)
        n_items = len(self.item2idx)

        # Positive interactions
        mask = filtered_df["user_id"].isin(self.user2idx) & filtered_df["video_id"].isin(self.item2idx)
        u_idx = filtered_df.loc[mask, "user_id"].map(self.user2idx).values.astype(np.int64)
        i_idx = filtered_df.loc[mask, "video_id"].map(self.item2idx).values.astype(np.int64)

        # Per-user positive items
        user_pos: Dict[int, Set[int]] = {}
        for u, i in zip(u_idx, i_idx):
            user_pos.setdefault(int(u), set()).add(int(i))

        # Per-user candidate items (for constrained negative sampling)
        user_candidates: Optional[Dict[int, List[int]]] = None
        if candidate_sets is not None:
            user_candidates = {}
            for uid, items in candidate_sets.items():
                if uid in self.user2idx:
                    uidx = self.user2idx[uid]
                    user_candidates[uidx] = [
                        self.item2idx[v] for v in items if v in self.item2idx
                    ]

        # Model
        self.model = BPRMFModel(n_users, n_items, self.embedding_dim).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr,
                                     weight_decay=self.weight_decay)

        n_interactions = len(u_idx)
        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            t0 = time.time()

            perm = np.random.permutation(n_interactions)
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, n_interactions, self.batch_size):
                idx = perm[start:start + self.batch_size]
                batch_users = torch.LongTensor(u_idx[idx]).to(self.device)
                batch_pos = torch.LongTensor(i_idx[idx]).to(self.device)

                # Negative sampling
                neg_items = []
                for u in u_idx[idx]:
                    u_int = int(u)
                    pos_set = user_pos.get(u_int, set())
                    if user_candidates and u_int in user_candidates:
                        # Sample from candidates minus positives
                        cands = [c for c in user_candidates[u_int] if c not in pos_set]
                        if cands:
                            neg_items.append(cands[np.random.randint(len(cands))])
                        else:
                            neg_items.append(np.random.randint(0, n_items))
                    else:
                        while True:
                            neg = np.random.randint(0, n_items)
                            if neg not in pos_set:
                                break
                        neg_items.append(neg)

                batch_neg = torch.LongTensor(neg_items).to(self.device)

                loss = self.model.bpr_loss(batch_users, batch_pos, batch_neg)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            elapsed = time.time() - t0

            if epoch % 5 == 0 or epoch == 1:
                log(f"  Epoch {epoch:3d}/{self.epochs}  loss={avg_loss:.4f}  "
                    f"time={elapsed:.1f}s")

            if avg_loss < best_loss - 1e-4:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    log(f"  Early stopping at epoch {epoch} (patience={self.patience})")
                    break

        log(f"BPR-MF trained: {n_users} users, {n_items} items, dim={self.embedding_dim}")

        if log_path:
            with open(log_path, "w") as f:
                f.write("\n".join(log_lines))

    @torch.no_grad()
    def rerank(self, user_id: int, candidate_video_ids: List[int],
               top_k: int = 20) -> List[int]:
        """
        Rerank candidate videos for a user. Returns top_k video_ids sorted by score.
        """
        if user_id not in self.user2idx:
            return candidate_video_ids[:top_k]

        self.model.eval()
        uidx = self.user2idx[user_id]

        valid = [(vid, self.item2idx[vid]) for vid in candidate_video_ids
                 if vid in self.item2idx]
        if not valid:
            return candidate_video_ids[:top_k]

        vids, iidxs = zip(*valid)
        user_t = torch.LongTensor([uidx] * len(iidxs)).to(self.device)
        item_t = torch.LongTensor(list(iidxs)).to(self.device)

        scores = self.model(user_t, item_t).cpu().numpy()
        ranked_idx = np.argsort(-scores)[:top_k]
        return [vids[i] for i in ranked_idx]

    @torch.no_grad()
    def rerank_batch(self, retrieval_results: Dict[int, List[int]],
                     top_k: int = 20) -> Dict[int, List[int]]:
        """Rerank retrieval results for all users."""
        ranked = {}
        for uid, candidates in retrieval_results.items():
            ranked[uid] = self.rerank(uid, candidates, top_k=top_k)
        return ranked

    def save(self, path: str):
        import pickle, os
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        state = {
            "model_state": self.model.state_dict() if self.model else None,
            "user2idx": self.user2idx, "idx2user": self.idx2user,
            "item2idx": self.item2idx, "idx2item": self.idx2item,
            "config": {"embedding_dim": self.embedding_dim},
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)
        print(f"BPR-MF saved to {path}")

    @staticmethod
    def load(path: str) -> "BPRRanker":
        import pickle
        with open(path, "rb") as f:
            state = pickle.load(f)
        obj = BPRRanker(embedding_dim=state["config"]["embedding_dim"])
        obj.user2idx = state["user2idx"]
        obj.idx2user = state["idx2user"]
        obj.item2idx = state["item2idx"]
        obj.idx2item = state["idx2item"]
        obj.model = BPRMFModel(len(obj.user2idx), len(obj.item2idx),
                                state["config"]["embedding_dim"])
        obj.model.load_state_dict(state["model_state"])
        return obj
