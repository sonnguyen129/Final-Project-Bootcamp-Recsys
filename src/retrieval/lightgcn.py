"""
LightGCN retrieval model — pure PyTorch implementation.

Reference: He et al., "LightGCN: Simplifying and Powering Graph Convolution
Network for Recommendation", SIGIR 2020.

Key idea: No feature transformation or nonlinear activation in GCN layers.
Final embedding = mean of all layer embeddings (layer 0 to L).
Trained with BPR loss on user-item bipartite graph.

Usage
-----
from src.retrieval.lightgcn import LightGCNRetriever
model = LightGCNRetriever()
model.fit(train_df, val_df=val_df)
user_embs, item_embs, user_ids, item_ids = model.get_all_embeddings()
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.sparse import coo_matrix
from typing import Dict, List, Optional, Tuple
import time

from src.data.preprocessing import build_id_maps


# Defaults
EMBEDDING_DIM = 128
N_LAYERS = 3
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
BATCH_SIZE = 4096
EPOCHS = 150
PATIENCE = 15
SEED = 42


class LightGCNModel(nn.Module):
    def __init__(self, n_users: int, n_items: int,
                 embedding_dim: int = EMBEDDING_DIM,
                 n_layers: int = N_LAYERS):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_layers = n_layers

        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)

        # Xavier init
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def compute_graph_embeddings(self, adj: torch.sparse.FloatTensor
                                  ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Multi-layer light graph convolution.
        adj: normalized adjacency matrix of shape (n_users+n_items, n_users+n_items)
        Returns final user and item embeddings.
        """
        # Concatenate user and item embeddings as initial E^(0)
        ego = torch.cat([self.user_embedding.weight,
                         self.item_embedding.weight], dim=0)
        all_layers = [ego]

        x = ego
        for _ in range(self.n_layers):
            x = torch.sparse.mm(adj, x)
            all_layers.append(x)

        # Final embedding = mean of all layers
        stacked = torch.stack(all_layers, dim=0)  # (L+1, N, dim)
        final = stacked.mean(dim=0)               # (N, dim)

        user_emb = final[:self.n_users]
        item_emb = final[self.n_users:]
        return user_emb, item_emb

    def bpr_loss(self, user_emb: torch.Tensor, item_emb: torch.Tensor,
                 users: torch.Tensor, pos_items: torch.Tensor,
                 neg_items: torch.Tensor) -> torch.Tensor:
        u = user_emb[users]
        pos = item_emb[pos_items]
        neg = item_emb[neg_items]

        pos_scores = (u * pos).sum(dim=1)
        neg_scores = (u * neg).sum(dim=1)

        loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10).mean()

        # L2 regularization on initial embeddings (not propagated)
        reg = (self.user_embedding.weight[users].norm(2).pow(2) +
               self.item_embedding.weight[pos_items].norm(2).pow(2) +
               self.item_embedding.weight[neg_items].norm(2).pow(2)) / len(users)
        return loss + 1e-5 * reg


def _build_normalized_adj(n_users: int, n_items: int,
                           user_indices: np.ndarray,
                           item_indices: np.ndarray,
                           device: torch.device) -> torch.sparse.FloatTensor:
    """
    Build symmetric normalized adjacency matrix D^{-1/2} A D^{-1/2}
    for the user-item bipartite graph.
    """
    n = n_users + n_items

    # Bipartite edges: user->item and item->user
    row = np.concatenate([user_indices, item_indices + n_users])
    col = np.concatenate([item_indices + n_users, user_indices])
    data = np.ones(len(row), dtype=np.float32)

    adj = coo_matrix((data, (row, col)), shape=(n, n))

    # D^{-1/2}
    degree = np.array(adj.sum(axis=1)).flatten()
    d_inv_sqrt = np.where(degree > 0, np.power(degree, -0.5), 0.0)

    # Normalize: D^{-1/2} A D^{-1/2}
    adj_coo = adj.tocoo()
    values = d_inv_sqrt[adj_coo.row] * adj_coo.data * d_inv_sqrt[adj_coo.col]

    indices = torch.LongTensor(np.stack([adj_coo.row, adj_coo.col]))
    values = torch.FloatTensor(values)
    adj_sparse = torch.sparse_coo_tensor(indices, values, (n, n)).to(device)
    return adj_sparse


class LightGCNRetriever:
    def __init__(self,
                 embedding_dim: int = EMBEDDING_DIM,
                 n_layers: int = N_LAYERS,
                 lr: float = LEARNING_RATE,
                 weight_decay: float = WEIGHT_DECAY,
                 batch_size: int = BATCH_SIZE,
                 epochs: int = EPOCHS,
                 patience: int = PATIENCE,
                 device: str = "auto"):
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model: Optional[LightGCNModel] = None
        self.user2idx: Dict[int, int] = {}
        self.idx2user: Dict[int, int] = {}
        self.item2idx: Dict[int, int] = {}
        self.idx2item: Dict[int, int] = {}
        self.adj: Optional[torch.sparse.FloatTensor] = None

        # Cached final embeddings (after training)
        self._user_embs: Optional[np.ndarray] = None
        self._item_embs: Optional[np.ndarray] = None

    def fit(self, train_df, val_df=None, log_path: str = None):
        """
        Train LightGCN on interactions.
        train_df: DataFrame with user_id, video_id, watch_ratio columns.
        val_df: optional validation DataFrame for early stopping.
        log_path: optional path to write training log.
        """
        torch.manual_seed(SEED)
        np.random.seed(SEED)

        log_lines = []
        def log(msg):
            print(msg)
            log_lines.append(msg)

        log(f"LightGCN training on {self.device}")

        # Build ID maps
        self.user2idx, self.idx2user, self.item2idx, self.idx2item = \
            build_id_maps(train_df)
        n_users = len(self.user2idx)
        n_items = len(self.item2idx)

        # Build interaction arrays (deduplicate user-item pairs for graph)
        mask = train_df["user_id"].isin(self.user2idx) & train_df["video_id"].isin(self.item2idx)
        # Deduplicate: one edge per (user, item) pair — extract only needed columns
        edges = train_df.loc[mask, ["user_id", "video_id"]].drop_duplicates()
        user_indices = edges["user_id"].map(self.user2idx).values.astype(np.int64)
        item_indices = edges["video_id"].map(self.item2idx).values.astype(np.int64)
        del edges
        log(f"  Graph: {len(user_indices):,} unique edges")

        # Normalized adjacency
        self.adj = _build_normalized_adj(n_users, n_items,
                                          user_indices, item_indices, self.device)

        # Build per-user positive item sets (for negative sampling)
        user_pos = {}
        for u, i in zip(user_indices, item_indices):
            user_pos.setdefault(int(u), set()).add(int(i))

        # Model
        self.model = LightGCNModel(n_users, n_items,
                                    self.embedding_dim, self.n_layers).to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr,
                               weight_decay=self.weight_decay)

        # Training loop
        n_interactions = len(user_indices)
        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            t0 = time.time()

            # Shuffle interactions
            perm = np.random.permutation(n_interactions)
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, n_interactions, self.batch_size):
                idx = perm[start:start + self.batch_size]
                batch_users = torch.LongTensor(user_indices[idx]).to(self.device)
                batch_pos = torch.LongTensor(item_indices[idx]).to(self.device)

                # Negative sampling: random items not in user's positive set
                neg_items = []
                for u in user_indices[idx]:
                    while True:
                        neg = np.random.randint(0, n_items)
                        if neg not in user_pos.get(int(u), set()):
                            break
                    neg_items.append(neg)
                batch_neg = torch.LongTensor(neg_items).to(self.device)

                # Forward
                user_emb, item_emb = self.model.compute_graph_embeddings(self.adj)
                loss = self.model.bpr_loss(user_emb, item_emb,
                                           batch_users, batch_pos, batch_neg)

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

            # Early stopping on training loss (or val loss if provided)
            if avg_loss < best_loss - 1e-4:
                best_loss = avg_loss
                patience_counter = 0
                # Cache best embeddings
                self._cache_embeddings()
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    log(f"  Early stopping at epoch {epoch} (patience={self.patience})")
                    break

        # Final embedding cache
        if self._user_embs is None:
            self._cache_embeddings()

        log(f"LightGCN trained: {n_users} users, {n_items} items, "
            f"{self.n_layers} layers, dim={self.embedding_dim}")

        if log_path:
            with open(log_path, "w") as f:
                f.write("\n".join(log_lines))

    @torch.no_grad()
    def _cache_embeddings(self):
        self.model.eval()
        user_emb, item_emb = self.model.compute_graph_embeddings(self.adj)
        self._user_embs = user_emb.cpu().numpy()
        self._item_embs = item_emb.cpu().numpy()

    def get_all_embeddings(self) -> Tuple[np.ndarray, np.ndarray, List[int], List[int]]:
        """
        Returns (user_embeddings, item_embeddings, user_ids, item_ids).
        Embeddings are numpy float32 arrays.
        """
        user_ids = [self.idx2user[i] for i in range(len(self.idx2user))]
        item_ids = [self.idx2item[i] for i in range(len(self.idx2item))]
        return self._user_embs, self._item_embs, user_ids, item_ids

    def get_user_embeddings(self) -> Tuple[List[int], np.ndarray]:
        """Return (user_ids, embeddings) for FAISS-compatible eval."""
        user_ids = [self.idx2user[i] for i in range(len(self.idx2user))]
        return user_ids, self._user_embs

    def get_item_embeddings(self) -> Tuple[List[int], np.ndarray]:
        """Return (item_ids, embeddings) for FAISS indexing."""
        item_ids = [self.idx2item[i] for i in range(len(self.idx2item))]
        return item_ids, self._item_embs

    def save(self, path: str):
        import pickle, os
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        # Save lightweight: model state + mappings + cached embeddings
        state = {
            "model_state": self.model.state_dict() if self.model else None,
            "user2idx": self.user2idx,
            "idx2user": self.idx2user,
            "item2idx": self.item2idx,
            "idx2item": self.idx2item,
            "user_embs": self._user_embs,
            "item_embs": self._item_embs,
            "config": {
                "embedding_dim": self.embedding_dim,
                "n_layers": self.n_layers,
            },
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)
        print(f"LightGCN saved to {path}")

    @staticmethod
    def load(path: str) -> "LightGCNRetriever":
        import pickle
        with open(path, "rb") as f:
            state = pickle.load(f)
        obj = LightGCNRetriever(
            embedding_dim=state["config"]["embedding_dim"],
            n_layers=state["config"]["n_layers"],
        )
        obj.user2idx = state["user2idx"]
        obj.idx2user = state["idx2user"]
        obj.item2idx = state["item2idx"]
        obj.idx2item = state["idx2item"]
        obj._user_embs = state["user_embs"]
        obj._item_embs = state["item_embs"]
        return obj
