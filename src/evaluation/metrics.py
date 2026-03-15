"""
Ranking evaluation metrics: Recall@K and NDCG@K.

Usage
-----
from src.evaluation.metrics import evaluate_retrieval

scores = evaluate_retrieval(
    recommendations,   # dict  {user_id: [ranked video_id list]}
    ground_truth,      # dict  {user_id: set of positive video_ids}
    k_list=[10, 20, 50]
)
"""

import numpy as np
from typing import Dict, List, Set


def recall_at_k(recommended: List[int], relevant: Set[int], k: int) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for item in recommended[:k] if item in relevant)
    return hits / min(len(relevant), k)


def ndcg_at_k(recommended: List[int], relevant: Set[int], k: int) -> float:
    if not relevant:
        return 0.0
    dcg = 0.0
    for rank, item in enumerate(recommended[:k], start=1):
        if item in relevant:
            dcg += 1.0 / np.log2(rank + 1)
    # Ideal DCG: first min(|relevant|, k) positions all hit
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_retrieval(
    recommendations: Dict[int, List[int]],
    ground_truth: Dict[int, Set[int]],
    k_list: List[int] = [10, 20, 50],
) -> Dict[str, float]:
    """
    Compute mean Recall@K and NDCG@K over all users present in both dicts.

    Parameters
    ----------
    recommendations : {user_id: [ranked item list]}
    ground_truth    : {user_id: {positive item ids}}
    k_list          : list of cutoff values

    Returns
    -------
    dict of metric_name → mean score
    """
    common_users = set(recommendations) & set(ground_truth)
    if not common_users:
        raise ValueError("No common users between recommendations and ground_truth.")

    results = {f"Recall@{k}": [] for k in k_list}
    results.update({f"NDCG@{k}": [] for k in k_list})

    for user in common_users:
        rec  = recommendations[user]
        rel  = ground_truth[user]
        for k in k_list:
            results[f"Recall@{k}"].append(recall_at_k(rec, rel, k))
            results[f"NDCG@{k}"].append(ndcg_at_k(rec, rel, k))

    return {metric: float(np.mean(vals)) for metric, vals in results.items()}


def print_metrics(scores: Dict[str, float], model_name: str = "Model") -> None:
    print(f"\n{'='*45}")
    print(f"  {model_name}")
    print(f"{'='*45}")
    for metric, value in sorted(scores.items()):
        print(f"  {metric:<14} {value:.4f}")
    print(f"{'='*45}\n")
