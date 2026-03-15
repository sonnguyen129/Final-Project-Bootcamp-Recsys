"""
Data preprocessing for KuaiRec.
- Binary label creation (watch_ratio > 2.0)
- Chronological train/val/test splits on big_matrix
- User watch sequence generation (sorted by timestamp)
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "datasets" / "KuaiRec 2.0" / "data"

POSITIVE_THRESHOLD = 2.0   # watch_ratio > 2.0 → label = 1
ITEM2VEC_THRESHOLD = 0.5   # watch_ratio > 0.5 → include in sequences
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
# test = 1 - TRAIN_RATIO - VAL_RATIO = 0.15


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_big_matrix() -> pd.DataFrame:
    path = DATA_DIR / "big_matrix.csv"
    df = pd.read_csv(path)
    df["label"] = (df["watch_ratio"] > POSITIVE_THRESHOLD).astype(int)
    return df


def load_small_matrix() -> pd.DataFrame:
    path = DATA_DIR / "small_matrix.csv"
    df = pd.read_csv(path)
    df["label"] = (df["watch_ratio"] > POSITIVE_THRESHOLD).astype(int)
    return df


def load_user_features() -> pd.DataFrame:
    path = DATA_DIR / "user_features.csv"
    return pd.read_csv(path)


def load_user_features_raw() -> pd.DataFrame:
    path = Path(__file__).resolve().parents[2] / "datasets" / "user_features_raw.csv"
    return pd.read_csv(path)


def load_item_categories() -> pd.DataFrame:
    path = DATA_DIR / "item_categories.csv"
    df = pd.read_csv(path)
    # feat column is stored as string repr of list → parse it
    df["feat"] = df["feat"].apply(lambda x: eval(x) if isinstance(x, str) else x)
    return df


def load_item_daily_features() -> pd.DataFrame:
    path = DATA_DIR / "item_daily_features.csv"
    return pd.read_csv(path)


def load_caption_category() -> pd.DataFrame:
    path = DATA_DIR / "kuairec_caption_category.csv"
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Train / Val / Test split  (chronological on big_matrix)
# ---------------------------------------------------------------------------

def split_big_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Chronological split of big_matrix interactions.
    Returns (train, val, test) DataFrames.
    """
    df = df.sort_values("timestamp").reset_index(drop=True)
    n = len(df)
    train_end = int(n * TRAIN_RATIO)
    val_end   = int(n * (TRAIN_RATIO + VAL_RATIO))

    train = df.iloc[:train_end].copy()
    val   = df.iloc[train_end:val_end].copy()
    test  = df.iloc[val_end:].copy()

    print(f"Split sizes  train: {len(train):,}  val: {len(val):,}  test: {len(test):,}")
    return train, val, test


# ---------------------------------------------------------------------------
# User-item index mapping
# ---------------------------------------------------------------------------

def build_id_maps(df: pd.DataFrame) -> tuple[dict, dict, dict, dict]:
    """
    Build consecutive integer mappings for user_id and video_id.
    Returns (user2idx, idx2user, item2idx, idx2item).
    """
    users = sorted(df["user_id"].unique())
    items = sorted(df["video_id"].unique())
    user2idx = {u: i for i, u in enumerate(users)}
    item2idx = {v: i for i, v in enumerate(items)}
    idx2user = {i: u for u, i in user2idx.items()}
    idx2item = {i: v for v, i in item2idx.items()}
    return user2idx, idx2user, item2idx, idx2item


# ---------------------------------------------------------------------------
# Sparse user-item matrix (for ALS)
# ---------------------------------------------------------------------------

def build_sparse_matrix(df: pd.DataFrame, user2idx: dict, item2idx: dict,
                         value_col: str = "watch_ratio"):
    """
    Build scipy CSR matrix of shape (n_users, n_items).
    Values are watch_ratio (used as confidence weights for ALS).
    Only keeps rows whose user_id and video_id are in the provided mappings.
    """
    from scipy.sparse import csr_matrix

    df = df[df["user_id"].isin(user2idx) & df["video_id"].isin(item2idx)].copy()
    row = df["user_id"].map(user2idx).values
    col = df["video_id"].map(item2idx).values
    data = df[value_col].values.astype(np.float32)

    n_users = len(user2idx)
    n_items = len(item2idx)
    mat = csr_matrix((data, (row, col)), shape=(n_users, n_items))
    return mat


# ---------------------------------------------------------------------------
# Watch sequences (for Item2Vec, GRU4Rec, SASRec)
# ---------------------------------------------------------------------------

def build_watch_sequences(df: pd.DataFrame,
                           threshold: float = ITEM2VEC_THRESHOLD) -> dict[int, list[int]]:
    """
    For each user, return an ordered list of video_ids they watched positively
    (watch_ratio >= threshold), sorted by timestamp ascending.
    """
    pos = df[df["watch_ratio"] >= threshold].copy()
    pos = pos.sort_values(["user_id", "timestamp"])
    sequences = pos.groupby("user_id")["video_id"].apply(list).to_dict()
    print(f"Sequences built for {len(sequences):,} users "
          f"(threshold={threshold}, total interactions={len(pos):,})")
    return sequences


# ---------------------------------------------------------------------------
# Ground-truth dict for evaluation (small_matrix)
# ---------------------------------------------------------------------------

def build_ground_truth(small_df: pd.DataFrame,
                        threshold: float = POSITIVE_THRESHOLD) -> dict[int, set[int]]:
    """
    For each user in small_matrix, collect the set of positively-watched video_ids.
    """
    pos = small_df[small_df["watch_ratio"] > threshold]
    gt = pos.groupby("user_id")["video_id"].apply(set).to_dict()
    print(f"Ground truth built for {len(gt):,} users from small_matrix")
    return gt


# ---------------------------------------------------------------------------
# Item aggregate features (for Two-Tower / DeepFM)
# ---------------------------------------------------------------------------

def build_item_agg_features(item_daily: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate item_daily_features over all dates → one row per video_id.
    Numeric columns are summed; play_progress is averaged.
    """
    num_cols = item_daily.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c not in ("video_id", "date")]

    avg_cols  = ["play_progress"]
    sum_cols  = [c for c in num_cols if c not in avg_cols]

    agg_dict = {c: "sum" for c in sum_cols}
    agg_dict.update({c: "mean" for c in avg_cols})
    agg_dict["video_duration"] = "first"  # constant per video

    item_agg = item_daily.groupby("video_id").agg(agg_dict).reset_index()
    return item_agg
