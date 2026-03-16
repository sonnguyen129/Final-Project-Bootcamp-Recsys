"""
GraphSAGE retrieval model using PyTorch Geometric.

Inductive GNN that learns to aggregate neighbor embeddings on a user-item
bipartite graph. Edge weights = watch_ratio. Generalizes to unseen nodes
(cold-start friendly).

Reference: Hamilton et al., "Inductive Representation Learning on Large Graphs",
NeurIPS 2017.

Usage
-----
from src.retrieval.graphsage import GraphSAGERetriever
model = GraphSAGERetriever()
model.fit(train_df)
user_ids, user_embs = model.get_user_embeddings()
item_ids, item_embs = model.get_item_embeddings()
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from typing import Dict, List, Optional, Tuple
import time

from src.data.preprocessing import build_id_maps


EMBEDDING_DIM = 128
HIDDEN_DIM = 128
N_LAYERS = 2
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
BATCH_SIZE = 4096
EPOCHS = 100
PATIENCE = 15
NUM_NEIGHBORS = [15, 10]  # neighbors to sample per layer
SEED = 42


class GraphSAGEModel(nn.Module):
    def __init__(self, n_users: int, n_items: int,
                 embedding_dim: int = EMBEDDING_DIM,
                 hidden_dim: int = HIDDEN_DIM,
                 n_layers: int = N_LAYERS):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items

        # Learnable initial embeddings (since we don't have rich node features)
        self.node_embedding = nn.Embedding(n_users + n_items, embedding_dim)
        nn.init.xavier_uniform_(self.node_embedding.weight)

        # SAGEConv layers
        self.convs = nn.ModuleList()
        in_dim = embedding_dim
        for i in range(n_layers):
            out_dim = hidden_dim if i < n_layers - 1 else embedding_dim
            self.convs.append(SAGEConv(in_dim, out_dim))
            in_dim = out_dim

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.1, training=self.training)
        return x

    def get_initial_embeddings(self) -> torch.Tensor:
        return self.node_embedding.weight


class GraphSAGERetriever:
    def __init__(self,
                 embedding_dim: int = EMBEDDING_DIM,
                 hidden_dim: int = HIDDEN_DIM,
                 n_layers: int = N_LAYERS,
                 lr: float = LEARNING_RATE,
                 weight_decay: float = WEIGHT_DECAY,
                 batch_size: int = BATCH_SIZE,
                 epochs: int = EPOCHS,
                 patience: int = PATIENCE,
                 device: str = "auto"):
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
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

        self.model: Optional[GraphSAGEModel] = None
        self.user2idx: Dict[int, int] = {}
        self.idx2user: Dict[int, int] = {}
        self.item2idx: Dict[int, int] = {}
        self.idx2item: Dict[int, int] = {}

        self._user_embs: Optional[np.ndarray] = None
        self._item_embs: Optional[np.ndarray] = None

    def fit(self, train_df, val_df=None, log_path: str = None):
        torch.manual_seed(SEED)
        np.random.seed(SEED)

        log_lines = []
        def log(msg):
            print(msg)
            log_lines.append(msg)

        log(f"GraphSAGE training on {self.device}")

        # Build ID maps
        self.user2idx, self.idx2user, self.item2idx, self.idx2item = \
            build_id_maps(train_df)
        n_users = len(self.user2idx)
        n_items = len(self.item2idx)
        n_nodes = n_users + n_items

        # Build edge index (bipartite: users 0..n_users-1, items n_users..n_nodes-1)
        mask = train_df["user_id"].isin(self.user2idx) & train_df["video_id"].isin(self.item2idx)
        # Deduplicate: one edge per (user, item) pair, keep mean watch_ratio
        edges = train_df.loc[mask, ["user_id", "video_id", "watch_ratio"]].groupby(
            ["user_id", "video_id"])["watch_ratio"].mean().reset_index()
        u_idx = edges["user_id"].map(self.user2idx).values.astype(np.int64)
        i_idx = (edges["video_id"].map(self.item2idx).values + n_users).astype(np.int64)
        wr = edges["watch_ratio"].values.astype(np.float32)
        log(f"  Graph: {len(edges):,} unique edges")
        del edges

        # Bidirectional edges
        src = np.concatenate([u_idx, i_idx])
        dst = np.concatenate([i_idx, u_idx])
        weights = np.concatenate([wr, wr])

        edge_index = torch.LongTensor(np.stack([src, dst]))
        edge_weight = torch.FloatTensor(weights)

        # Per-user positive sets for negative sampling
        user_pos = {}
        for u, i in zip(u_idx, i_idx - n_users):
            user_pos.setdefault(int(u), set()).add(int(i))

        # Model
        self.model = GraphSAGEModel(n_users, n_items,
                                     self.embedding_dim, self.hidden_dim,
                                     self.n_layers).to(self.device)

        edge_index = edge_index.to(self.device)
        edge_weight = edge_weight.to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr,
                                     weight_decay=self.weight_decay)

        # Training with full-batch graph convolution + mini-batch BPR
        n_interactions = len(u_idx)
        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            t0 = time.time()

            # Full forward pass to get all embeddings
            x = self.model.get_initial_embeddings().to(self.device)
            all_embs = self.model(x, edge_index, edge_weight)
            user_emb = all_embs[:n_users]
            item_emb = all_embs[n_users:]

            # Mini-batch BPR loss
            perm = np.random.permutation(n_interactions)
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, n_interactions, self.batch_size):
                idx = perm[start:start + self.batch_size]
                batch_users = torch.LongTensor(u_idx[idx]).to(self.device)
                # i_idx already has +n_users offset, subtract to get item indices
                batch_pos = torch.LongTensor(
                    (i_idx[idx] - n_users).astype(np.int64)
                ).to(self.device)

                # Negative sampling
                neg_items = []
                for u in u_idx[idx]:
                    while True:
                        neg = np.random.randint(0, n_items)
                        if neg not in user_pos.get(int(u), set()):
                            break
                    neg_items.append(neg)
                batch_neg = torch.LongTensor(neg_items).to(self.device)

                u_e = user_emb[batch_users]
                pos_e = item_emb[batch_pos]
                neg_e = item_emb[batch_neg]

                pos_scores = (u_e * pos_e).sum(dim=1)
                neg_scores = (u_e * neg_e).sum(dim=1)
                loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10).mean()

                # L2 reg
                reg = (u_e.norm(2).pow(2) + pos_e.norm(2).pow(2) +
                       neg_e.norm(2).pow(2)) / len(batch_users) * 1e-5
                loss = loss + reg

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
                self._cache_embeddings(edge_index, edge_weight)
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    log(f"  Early stopping at epoch {epoch} (patience={self.patience})")
                    break

        if self._user_embs is None:
            self._cache_embeddings(edge_index, edge_weight)

        log(f"GraphSAGE trained: {n_users} users, {n_items} items, "
            f"{self.n_layers} layers, dim={self.embedding_dim}")

        if log_path:
            with open(log_path, "w") as f:
                f.write("\n".join(log_lines))

    @torch.no_grad()
    def _cache_embeddings(self, edge_index, edge_weight):
        self.model.eval()
        n_users = len(self.user2idx)
        x = self.model.get_initial_embeddings().to(self.device)
        all_embs = self.model(x, edge_index, edge_weight)
        self._user_embs = all_embs[:n_users].cpu().numpy()
        self._item_embs = all_embs[n_users:].cpu().numpy()

    def get_user_embeddings(self) -> Tuple[List[int], np.ndarray]:
        user_ids = [self.idx2user[i] for i in range(len(self.idx2user))]
        return user_ids, self._user_embs

    def get_item_embeddings(self) -> Tuple[List[int], np.ndarray]:
        item_ids = [self.idx2item[i] for i in range(len(self.idx2item))]
        return item_ids, self._item_embs

    def get_all_embeddings(self) -> Tuple[np.ndarray, np.ndarray, List[int], List[int]]:
        user_ids = [self.idx2user[i] for i in range(len(self.idx2user))]
        item_ids = [self.idx2item[i] for i in range(len(self.idx2item))]
        return self._user_embs, self._item_embs, user_ids, item_ids

    def save(self, path: str):
        import pickle, os
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        state = {
            "model_state": self.model.state_dict() if self.model else None,
            "user2idx": self.user2idx, "idx2user": self.idx2user,
            "item2idx": self.item2idx, "idx2item": self.idx2item,
            "user_embs": self._user_embs, "item_embs": self._item_embs,
            "config": {
                "embedding_dim": self.embedding_dim,
                "hidden_dim": self.hidden_dim,
                "n_layers": self.n_layers,
            },
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)
        print(f"GraphSAGE saved to {path}")

    @staticmethod
    def load(path: str) -> "GraphSAGERetriever":
        import pickle
        with open(path, "rb") as f:
            state = pickle.load(f)
        cfg = state["config"]
        obj = GraphSAGERetriever(
            embedding_dim=cfg["embedding_dim"],
            hidden_dim=cfg["hidden_dim"],
            n_layers=cfg["n_layers"],
        )
        obj.user2idx = state["user2idx"]
        obj.idx2user = state["idx2user"]
        obj.item2idx = state["item2idx"]
        obj.idx2item = state["idx2item"]
        obj._user_embs = state["user_embs"]
        obj._item_embs = state["item_embs"]
        return obj
