"""
Shared FAISS-based retrieval evaluation.

All retrieval models produce embeddings → this module handles:
1. Filter items to candidate pool (small_matrix videos)
2. Build FAISS index
3. Search for each eval user
4. Compute Recall@K & NDCG@K

Usage
-----
from src.evaluation.retrieval_eval import evaluate_via_faiss

scores, recs = evaluate_via_faiss(
    item_ids, item_embs, user_ids, user_embs,
    ground_truth, candidate_pool, model_name="LightGCN"
)
"""

import numpy as np
from typing import Dict, List, Set, Tuple

from src.indexing.faiss_index import FAISSIndex
from src.evaluation.metrics import evaluate_retrieval, print_metrics


K_LIST_DEFAULT = [10, 20, 50, 100]


def evaluate_via_faiss(
    item_ids: List[int],
    item_embs: np.ndarray,
    user_ids: List[int],
    user_embs: np.ndarray,
    ground_truth: Dict[int, Set[int]],
    candidate_pool: Set[int],
    model_name: str = "Model",
    k_list: List[int] = None,
    verbose: bool = True,
) -> Tuple[Dict[str, float], Dict[int, List[int]]]:
    """
    Evaluate a retrieval model using FAISS on a restricted candidate pool.

    Parameters
    ----------
    item_ids : list of video_ids from the model
    item_embs : np.ndarray (n_items, dim)
    user_ids : list of user_ids (aligned with user_embs rows)
    user_embs : np.ndarray (n_users, dim)
    ground_truth : dict {user_id: set(positive_video_ids)}
    candidate_pool : set of video_ids to restrict FAISS index to
    model_name : str for display
    k_list : list of K values for metrics
    verbose : print metrics

    Returns
    -------
    (scores_dict, recommendations_dict)
    """
    if k_list is None:
        k_list = K_LIST_DEFAULT

    # Filter items to candidate pool
    pool_mask = [i for i, vid in enumerate(item_ids) if vid in candidate_pool]
    pool_item_ids = [item_ids[i] for i in pool_mask]
    pool_item_embs = item_embs[pool_mask]

    if verbose:
        print(f"[{model_name}] FAISS index: {len(pool_item_ids):,} / "
              f"{len(item_ids):,} items in candidate pool")

    if len(pool_item_ids) == 0:
        print(f"[{model_name}] WARNING: No items in candidate pool!")
        return {}, {}

    # Build FAISS index
    index = FAISSIndex(mode="exact")
    index.build(pool_item_ids, pool_item_embs)

    # Map user_ids to embedding rows
    user_id_to_idx = {uid: i for i, uid in enumerate(user_ids)}

    # Search for each eval user (users in both user_ids and ground_truth)
    eval_users = [uid for uid in user_ids if uid in ground_truth]
    max_k = max(k_list)

    recommendations = {}
    for uid in eval_users:
        idx = user_id_to_idx[uid]
        user_vec = user_embs[idx]
        recommendations[uid] = index.search(user_vec, k=max_k)

    # Compute metrics
    scores = evaluate_retrieval(recommendations, ground_truth, k_list)
    if verbose:
        print_metrics(scores, model_name)

    return scores, recommendations
