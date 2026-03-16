"""
Two-Tower retrieval model — custom PyTorch implementation.

User tower: demographics + watch history aggregates → MLP → embedding
Item tower: categories + daily stats → MLP → embedding
Training: in-batch negatives with cosine similarity + cross-entropy loss.

This is the only retrieval model that uses side features, making it
cold-start friendly.

Usage
-----
from src.retrieval.two_tower import TwoTowerRetriever
model = TwoTowerRetriever()
model.fit(train_df, user_features_df, item_features_df)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import LabelEncoder, StandardScaler
import time

from src.data.preprocessing import (
    build_id_maps, load_user_features_raw, load_item_categories,
    load_item_daily_features, build_item_agg_features, build_watch_sequences,
)


EMBEDDING_DIM = 128
HIDDEN_DIM = 256
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
BATCH_SIZE = 4096
EPOCHS = 50
PATIENCE = 10
TEMPERATURE = 0.05   # for in-batch softmax
SEED = 42


class UserTower(nn.Module):
    def __init__(self, input_dim: int, embedding_dim: int = EMBEDDING_DIM,
                 hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, embedding_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


class ItemTower(nn.Module):
    def __init__(self, input_dim: int, embedding_dim: int = EMBEDDING_DIM,
                 hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, embedding_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


class TwoTowerModel(nn.Module):
    def __init__(self, user_input_dim: int, item_input_dim: int,
                 embedding_dim: int = EMBEDDING_DIM, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.user_tower = UserTower(user_input_dim, embedding_dim, hidden_dim)
        self.item_tower = ItemTower(item_input_dim, embedding_dim, hidden_dim)

    def forward(self, user_features, item_features):
        user_emb = self.user_tower(user_features)
        item_emb = self.item_tower(item_features)
        return user_emb, item_emb


def _prepare_user_features(train_df, user2idx) -> Tuple[np.ndarray, int]:
    """
    Build user feature matrix from demographics + watch history aggregates.
    Returns (feature_matrix [n_users, feat_dim], feat_dim).
    """
    # Load raw user features
    try:
        user_raw = load_user_features_raw()
    except FileNotFoundError:
        user_raw = pd.DataFrame({"user_id": list(user2idx.keys())})

    # Encode categorical columns
    cat_cols = ["gender", "age_range", "fre_city_level", "user_active_degree"]
    available_cats = [c for c in cat_cols if c in user_raw.columns]
    encoders = {}
    for col in available_cats:
        user_raw[col] = user_raw[col].fillna("UNKNOWN").astype(str)
        le = LabelEncoder()
        user_raw[col + "_enc"] = le.fit_transform(user_raw[col])
        encoders[col] = le

    # Numeric cols
    num_cols = ["mod_price"]
    available_nums = [c for c in num_cols if c in user_raw.columns]
    for col in available_nums:
        user_raw[col] = pd.to_numeric(user_raw[col], errors="coerce").fillna(0).astype(float)

    # Watch history aggregates per user from train_df
    user_agg = train_df.groupby("user_id").agg(
        n_interactions=("video_id", "count"),
        mean_watch_ratio=("watch_ratio", "mean"),
        std_watch_ratio=("watch_ratio", "std"),
        n_unique_videos=("video_id", "nunique"),
    ).reset_index()
    user_agg["std_watch_ratio"] = user_agg["std_watch_ratio"].fillna(0)

    # Merge
    feat_cols_enc = [c + "_enc" for c in available_cats]
    feat_cols = feat_cols_enc + available_nums

    user_feat_df = pd.DataFrame({"user_id": list(user2idx.keys())})
    user_feat_df = user_feat_df.merge(user_raw[["user_id"] + feat_cols],
                                       on="user_id", how="left")
    user_feat_df = user_feat_df.merge(user_agg, on="user_id", how="left")

    # Fill NaN
    for c in user_feat_df.columns:
        if c != "user_id":
            user_feat_df[c] = user_feat_df[c].fillna(0)

    # Order by idx
    user_feat_df["_idx"] = user_feat_df["user_id"].map(user2idx)
    user_feat_df = user_feat_df.sort_values("_idx")

    feature_cols = [c for c in user_feat_df.columns if c not in ("user_id", "_idx")]
    mat = user_feat_df[feature_cols].values.astype(np.float32)

    # Normalize
    scaler = StandardScaler()
    mat = scaler.fit_transform(mat)

    return mat.astype(np.float32), mat.shape[1]


def _prepare_item_features(train_df, item2idx) -> Tuple[np.ndarray, int]:
    """
    Build item feature matrix from categories + daily stats.
    Returns (feature_matrix [n_items, feat_dim], feat_dim).
    """
    n_items = len(item2idx)

    # Item categories (multi-hot encoding)
    try:
        cats_df = load_item_categories()
        all_cats = set()
        for feats in cats_df["feat"]:
            if isinstance(feats, list):
                all_cats.update(feats)
        cat_list = sorted(all_cats)
        cat2idx = {c: i for i, c in enumerate(cat_list)}
        n_cats = len(cat_list)

        cat_matrix = np.zeros((n_items, n_cats), dtype=np.float32)
        for _, row in cats_df.iterrows():
            vid = row["video_id"]
            if vid in item2idx and isinstance(row["feat"], list):
                for c in row["feat"]:
                    if c in cat2idx:
                        cat_matrix[item2idx[vid], cat2idx[c]] = 1.0
    except (FileNotFoundError, KeyError):
        cat_matrix = np.zeros((n_items, 1), dtype=np.float32)

    # Item daily features (aggregated)
    try:
        daily = load_item_daily_features()
        item_agg = build_item_agg_features(daily)

        # Select numeric columns
        agg_num_cols = [c for c in item_agg.columns
                        if c != "video_id" and item_agg[c].dtype in [np.float64, np.float32, np.int64]]
        agg_num_cols = agg_num_cols[:20]  # Limit to top 20 features

        agg_matrix = np.zeros((n_items, len(agg_num_cols)), dtype=np.float32)
        for _, row in item_agg.iterrows():
            vid = row["video_id"]
            if vid in item2idx:
                agg_matrix[item2idx[vid]] = row[agg_num_cols].values.astype(np.float32)

        # Normalize
        scaler = StandardScaler()
        agg_matrix = scaler.fit_transform(agg_matrix)
    except (FileNotFoundError, KeyError):
        agg_matrix = np.zeros((n_items, 1), dtype=np.float32)

    # Concatenate
    feat_matrix = np.concatenate([cat_matrix, agg_matrix], axis=1).astype(np.float32)
    return feat_matrix, feat_matrix.shape[1]


class TwoTowerRetriever:
    def __init__(self,
                 embedding_dim: int = EMBEDDING_DIM,
                 hidden_dim: int = HIDDEN_DIM,
                 lr: float = LEARNING_RATE,
                 weight_decay: float = WEIGHT_DECAY,
                 batch_size: int = BATCH_SIZE,
                 epochs: int = EPOCHS,
                 patience: int = PATIENCE,
                 temperature: float = TEMPERATURE,
                 device: str = "auto"):
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.temperature = temperature

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model: Optional[TwoTowerModel] = None
        self.user2idx: Dict[int, int] = {}
        self.idx2user: Dict[int, int] = {}
        self.item2idx: Dict[int, int] = {}
        self.idx2item: Dict[int, int] = {}

        self._user_feats: Optional[np.ndarray] = None
        self._item_feats: Optional[np.ndarray] = None
        self._user_embs: Optional[np.ndarray] = None
        self._item_embs: Optional[np.ndarray] = None

    def fit(self, train_df, log_path: str = None):
        torch.manual_seed(SEED)
        np.random.seed(SEED)

        log_lines = []
        def log(msg):
            print(msg)
            log_lines.append(msg)

        log(f"Two-Tower training on {self.device}")

        # Build ID maps
        self.user2idx, self.idx2user, self.item2idx, self.idx2item = \
            build_id_maps(train_df)

        # Prepare features
        log("  Preparing user features...")
        self._user_feats, user_feat_dim = _prepare_user_features(train_df, self.user2idx)
        log(f"  User features: {self._user_feats.shape}")

        log("  Preparing item features...")
        self._item_feats, item_feat_dim = _prepare_item_features(train_df, self.item2idx)
        log(f"  Item features: {self._item_feats.shape}")

        # Model
        self.model = TwoTowerModel(user_feat_dim, item_feat_dim,
                                    self.embedding_dim, self.hidden_dim).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr,
                                     weight_decay=self.weight_decay)

        # Build positive pairs (no .copy() — just extract arrays)
        mask = train_df["user_id"].isin(self.user2idx) & train_df["video_id"].isin(self.item2idx)
        u_idx = train_df.loc[mask, "user_id"].map(self.user2idx).values.astype(np.int64)
        i_idx = train_df.loc[mask, "video_id"].map(self.item2idx).values.astype(np.int64)

        user_feats_t = torch.FloatTensor(self._user_feats).to(self.device)
        item_feats_t = torch.FloatTensor(self._item_feats).to(self.device)

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
                batch_u = u_idx[idx]
                batch_i = i_idx[idx]

                # Get features
                u_feat = user_feats_t[batch_u]
                i_feat = item_feats_t[batch_i]

                # Forward through towers
                u_emb, i_emb = self.model(u_feat, i_feat)

                # In-batch negatives: similarity matrix (batch x batch)
                sim_matrix = torch.mm(u_emb, i_emb.t()) / self.temperature

                # Labels: diagonal is positive
                labels = torch.arange(len(batch_u), device=self.device)
                loss = F.cross_entropy(sim_matrix, labels)

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
                self._cache_embeddings()
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    log(f"  Early stopping at epoch {epoch} (patience={self.patience})")
                    break

        if self._user_embs is None:
            self._cache_embeddings()

        n_users = len(self.user2idx)
        n_items = len(self.item2idx)
        log(f"Two-Tower trained: {n_users} users, {n_items} items, dim={self.embedding_dim}")

        if log_path:
            with open(log_path, "w") as f:
                f.write("\n".join(log_lines))

    @torch.no_grad()
    def _cache_embeddings(self):
        self.model.eval()
        user_feats_t = torch.FloatTensor(self._user_feats).to(self.device)
        item_feats_t = torch.FloatTensor(self._item_feats).to(self.device)

        # Process in chunks to avoid OOM
        chunk = 8192
        user_embs = []
        for i in range(0, len(self._user_feats), chunk):
            emb = self.model.user_tower(user_feats_t[i:i + chunk])
            user_embs.append(emb.cpu().numpy())
        self._user_embs = np.concatenate(user_embs, axis=0)

        item_embs = []
        for i in range(0, len(self._item_feats), chunk):
            emb = self.model.item_tower(item_feats_t[i:i + chunk])
            item_embs.append(emb.cpu().numpy())
        self._item_embs = np.concatenate(item_embs, axis=0)

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
            "user_feats": self._user_feats, "item_feats": self._item_feats,
            "config": {
                "embedding_dim": self.embedding_dim,
                "hidden_dim": self.hidden_dim,
            },
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)
        print(f"Two-Tower saved to {path}")

    @staticmethod
    def load(path: str) -> "TwoTowerRetriever":
        import pickle
        with open(path, "rb") as f:
            state = pickle.load(f)
        cfg = state["config"]
        obj = TwoTowerRetriever(
            embedding_dim=cfg["embedding_dim"],
            hidden_dim=cfg["hidden_dim"],
        )
        obj.user2idx = state["user2idx"]
        obj.idx2user = state["idx2user"]
        obj.item2idx = state["item2idx"]
        obj.idx2item = state["idx2item"]
        obj._user_embs = state["user_embs"]
        obj._item_embs = state["item_embs"]
        obj._user_feats = state["user_feats"]
        obj._item_feats = state["item_feats"]
        return obj
