"""
Week 2 Training Script — Advanced Retrieval + Ranking Baseline

Trains LightGCN, GraphSAGE, Two-Tower retrieval models and BPR-MF ranker.
Evaluates all 5 retrieval models (including Week 1 ALS + Item2Vec) via FAISS.
Runs end-to-end retrieval→ranking pipeline with BPR-MF.

Logs all output to experiments/week2_train.log

Usage
-----
conda activate stock
python scripts/train_week2.py
"""

import sys
import os
import gc
import time
import logging
import numpy as np
import pandas as pd
import torch

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.preprocessing import (
    load_big_matrix, load_small_matrix, split_big_matrix,
    build_adaptive_ground_truth, build_ground_truth,
)
from src.indexing.faiss_index import FAISSIndex
from src.evaluation.metrics import evaluate_retrieval, print_metrics

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "experiments")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_PATH = os.path.join(LOG_DIR, "week2_train.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODELS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Shared evaluation function (same protocol as Week 1)
# ---------------------------------------------------------------------------
def evaluate_via_faiss(item_ids, item_embs, user_ids, user_embs,
                       ground_truth, candidate_pool, model_name,
                       k_list=[10, 20, 50, 100]):
    """
    Evaluate a retrieval model using FAISS on the small_matrix candidate pool.

    Parameters
    ----------
    item_ids : list of video_ids from the model
    item_embs : np.ndarray (n_items, dim)
    user_ids : list of user_ids
    user_embs : np.ndarray (n_users, dim)
    ground_truth : dict {user_id: set(positive_video_ids)}
    candidate_pool : set of video_ids to restrict FAISS index to
    model_name : str
    k_list : list of K values

    Returns
    -------
    dict of metric scores, dict of {user_id: [ranked video_ids]}
    """
    # Filter items to candidate pool
    pool_mask = [i for i, vid in enumerate(item_ids) if vid in candidate_pool]
    pool_item_ids = [item_ids[i] for i in pool_mask]
    pool_item_embs = item_embs[pool_mask]

    log.info(f"[{model_name}] {len(pool_item_ids)}/{len(item_ids)} items in candidate pool")

    if len(pool_item_ids) == 0:
        log.warning(f"[{model_name}] No items in candidate pool!")
        return {}, {}

    # Build FAISS index on pool items
    index = FAISSIndex(mode="exact")
    index.build(pool_item_ids, pool_item_embs)

    # Evaluate each user
    eval_users = [uid for uid in user_ids if uid in ground_truth]
    user_id_to_idx = {uid: i for i, uid in enumerate(user_ids)}

    recommendations = {}
    max_k = max(k_list)
    for uid in eval_users:
        idx = user_id_to_idx[uid]
        user_vec = user_embs[idx]
        recs = index.search(user_vec, k=max_k)
        recommendations[uid] = recs

    scores = evaluate_retrieval(recommendations, ground_truth, k_list)
    print_metrics(scores, model_name)
    return scores, recommendations


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    total_start = time.time()

    # ===== Load data =====
    log.info("Loading data...")
    big_matrix = load_big_matrix()
    small_matrix = load_small_matrix()
    train_df, val_df, test_df = split_big_matrix(big_matrix)
    del big_matrix, val_df, test_df
    gc.collect()

    # Ground truth
    gt_adaptive = build_adaptive_ground_truth(small_matrix)
    gt_fixed = build_ground_truth(small_matrix)
    candidate_pool = set(small_matrix["video_id"].unique())
    log.info(f"Candidate pool: {len(candidate_pool)} videos")

    all_results = {}

    # ===== 1. Load Week 1 models (ALS + Item2Vec) =====
    log.info("\n" + "="*60)
    log.info("LOADING WEEK 1 MODELS")
    log.info("="*60)

    # ALS
    try:
        from src.retrieval.als import ALSRetriever
        als_path = os.path.join(MODELS_DIR, "als.pkl")
        if os.path.exists(als_path):
            als = ALSRetriever.load(als_path)
            als_item_ids = [als.idx2item[i] for i in range(len(als.idx2item))]
            als_item_embs = als.get_item_embeddings()
            als_user_ids = [als.idx2user[i] for i in range(len(als.idx2user))]
            als_user_embs = als.get_user_embeddings()

            scores, _ = evaluate_via_faiss(
                als_item_ids, als_item_embs,
                als_user_ids, als_user_embs,
                gt_adaptive, candidate_pool, "ALS (Week1)"
            )
            all_results["ALS"] = scores
        else:
            log.info("ALS model not found, retraining...")
            als = ALSRetriever()
            als.fit(train_df)
            als.save(als_path)

            als_item_ids = [als.idx2item[i] for i in range(len(als.idx2item))]
            als_item_embs = als.get_item_embeddings()
            als_user_ids = [als.idx2user[i] for i in range(len(als.idx2user))]
            als_user_embs = als.get_user_embeddings()

            scores, _ = evaluate_via_faiss(
                als_item_ids, als_item_embs,
                als_user_ids, als_user_embs,
                gt_adaptive, candidate_pool, "ALS"
            )
            all_results["ALS"] = scores
    except Exception as e:
        log.error(f"ALS failed: {e}")

    # Item2Vec
    try:
        from src.retrieval.item2vec import Item2VecRetriever
        i2v_path = os.path.join(MODELS_DIR, "item2vec.pkl")
        if os.path.exists(i2v_path):
            i2v = Item2VecRetriever.load(i2v_path)
            i2v_item_ids, i2v_item_embs = i2v.get_item_embeddings()
            i2v_user_ids = list(i2v.user_embeddings.keys())
            i2v_user_embs = np.stack([i2v.user_embeddings[u] for u in i2v_user_ids])

            scores, _ = evaluate_via_faiss(
                i2v_item_ids, i2v_item_embs,
                i2v_user_ids, i2v_user_embs,
                gt_adaptive, candidate_pool, "Item2Vec (Week1)"
            )
            all_results["Item2Vec"] = scores
        else:
            log.info("Item2Vec model not found, retraining...")
            i2v = Item2VecRetriever()
            i2v.fit(train_df)
            i2v.save(i2v_path)

            i2v_item_ids, i2v_item_embs = i2v.get_item_embeddings()
            i2v_user_ids = list(i2v.user_embeddings.keys())
            i2v_user_embs = np.stack([i2v.user_embeddings[u] for u in i2v_user_ids])

            scores, _ = evaluate_via_faiss(
                i2v_item_ids, i2v_item_embs,
                i2v_user_ids, i2v_user_embs,
                gt_adaptive, candidate_pool, "Item2Vec"
            )
            all_results["Item2Vec"] = scores
    except Exception as e:
        log.error(f"Item2Vec failed: {e}")

    # Free Week 1 model objects (keep only embedding arrays)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ===== 2. Train LightGCN =====
    log.info("\n" + "="*60)
    log.info("TRAINING LIGHTGCN")
    log.info("="*60)

    try:
        from src.retrieval.lightgcn import LightGCNRetriever
        lgcn = LightGCNRetriever(
            embedding_dim=128, n_layers=3,
            epochs=150, patience=15, batch_size=4096,
        )
        lgcn_log = os.path.join(LOG_DIR, "lightgcn_train.log")
        lgcn.fit(train_df, log_path=lgcn_log)
        lgcn.save(os.path.join(MODELS_DIR, "lightgcn.pkl"))

        lgcn_item_ids, lgcn_item_embs = lgcn.get_item_embeddings()
        lgcn_user_ids, lgcn_user_embs = lgcn.get_user_embeddings()

        scores, lgcn_recs = evaluate_via_faiss(
            lgcn_item_ids, lgcn_item_embs,
            lgcn_user_ids, lgcn_user_embs,
            gt_adaptive, candidate_pool, "LightGCN"
        )
        all_results["LightGCN"] = scores
    except Exception as e:
        log.error(f"LightGCN failed: {e}", exc_info=True)
        lgcn_recs = {}

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ===== 3. Train GraphSAGE =====
    log.info("\n" + "="*60)
    log.info("TRAINING GRAPHSAGE")
    log.info("="*60)

    try:
        from src.retrieval.graphsage import GraphSAGERetriever
        gsage = GraphSAGERetriever(
            embedding_dim=128, hidden_dim=128, n_layers=2,
            epochs=100, patience=15, batch_size=4096,
        )
        gsage_log = os.path.join(LOG_DIR, "graphsage_train.log")
        gsage.fit(train_df, log_path=gsage_log)
        gsage.save(os.path.join(MODELS_DIR, "graphsage.pkl"))

        gsage_item_ids, gsage_item_embs = gsage.get_item_embeddings()
        gsage_user_ids, gsage_user_embs = gsage.get_user_embeddings()

        scores, gsage_recs = evaluate_via_faiss(
            gsage_item_ids, gsage_item_embs,
            gsage_user_ids, gsage_user_embs,
            gt_adaptive, candidate_pool, "GraphSAGE"
        )
        all_results["GraphSAGE"] = scores
    except Exception as e:
        log.error(f"GraphSAGE failed: {e}", exc_info=True)
        gsage_recs = {}

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ===== 4. Train Two-Tower =====
    log.info("\n" + "="*60)
    log.info("TRAINING TWO-TOWER")
    log.info("="*60)

    try:
        from src.retrieval.two_tower import TwoTowerRetriever
        tt = TwoTowerRetriever(
            embedding_dim=128, hidden_dim=256,
            epochs=50, patience=10, batch_size=4096,
        )
        tt_log = os.path.join(LOG_DIR, "two_tower_train.log")
        tt.fit(train_df, log_path=tt_log)
        tt.save(os.path.join(MODELS_DIR, "two_tower.pkl"))

        tt_item_ids, tt_item_embs = tt.get_item_embeddings()
        tt_user_ids, tt_user_embs = tt.get_user_embeddings()

        scores, tt_recs = evaluate_via_faiss(
            tt_item_ids, tt_item_embs,
            tt_user_ids, tt_user_embs,
            gt_adaptive, candidate_pool, "Two-Tower"
        )
        all_results["Two-Tower"] = scores
    except Exception as e:
        log.error(f"Two-Tower failed: {e}", exc_info=True)
        tt_recs = {}

    # ===== 5. Comparison table =====
    log.info("\n" + "="*60)
    log.info("RETRIEVAL MODEL COMPARISON (Adaptive Ground Truth)")
    log.info("="*60)

    if all_results:
        metrics_order = ["NDCG@10", "NDCG@20", "NDCG@50", "NDCG@100",
                         "Recall@10", "Recall@20", "Recall@50", "Recall@100"]
        header = f"{'Model':<15}" + "".join(f"{m:>12}" for m in metrics_order)
        log.info(header)
        log.info("-" * len(header))
        for model_name, scores in all_results.items():
            row = f"{model_name:<15}"
            for m in metrics_order:
                row += f"{scores.get(m, 0.0):>12.4f}"
            log.info(row)

        # Save to CSV
        results_df = pd.DataFrame(all_results).T
        results_df.index.name = "Model"
        csv_path = os.path.join(LOG_DIR, "week2_retrieval_results.csv")
        results_df.to_csv(csv_path)
        log.info(f"\nResults saved to {csv_path}")

    # ===== 6. Select best retrieval model & update FAISS =====
    if all_results:
        best_model = max(all_results, key=lambda m: all_results[m].get("NDCG@10", 0))
        log.info(f"\nBest retrieval model: {best_model} (NDCG@10={all_results[best_model]['NDCG@10']:.4f})")

        # Update FAISS index with best model embeddings
        best_retrievers = {
            "ALS": lambda: (als_item_ids, als_item_embs) if "ALS" in all_results else None,
            "Item2Vec": lambda: (i2v_item_ids, i2v_item_embs) if "Item2Vec" in all_results else None,
            "LightGCN": lambda: (lgcn_item_ids, lgcn_item_embs) if "LightGCN" in all_results else None,
            "GraphSAGE": lambda: (gsage_item_ids, gsage_item_embs) if "GraphSAGE" in all_results else None,
            "Two-Tower": lambda: (tt_item_ids, tt_item_embs) if "Two-Tower" in all_results else None,
        }
        best_data = best_retrievers.get(best_model, lambda: None)()
        if best_data:
            best_item_ids, best_item_embs = best_data
            # Build and save FAISS index
            faiss_exact = FAISSIndex(mode="exact")
            faiss_exact.build(best_item_ids, best_item_embs)
            faiss_exact.save(
                os.path.join(MODELS_DIR, f"faiss_{best_model.lower().replace('-', '_')}_exact.index"),
                os.path.join(MODELS_DIR, f"faiss_{best_model.lower().replace('-', '_')}_exact_meta.json"),
            )
            log.info(f"FAISS index updated with {best_model} embeddings")

    # ===== 7. Train BPR-MF ranker on best retrieval candidates =====
    log.info("\n" + "="*60)
    log.info("TRAINING BPR-MF RANKER")
    log.info("="*60)

    # Use LightGCN retrieval results if available, else best available
    retrieval_recs = lgcn_recs if lgcn_recs else (gsage_recs if gsage_recs else tt_recs)

    if retrieval_recs:
        try:
            from src.ranking.bpr import BPRRanker

            bpr = BPRRanker(embedding_dim=64, epochs=80, patience=10)
            bpr_log = os.path.join(LOG_DIR, "bpr_train.log")
            bpr.fit(train_df, candidate_sets=retrieval_recs, log_path=bpr_log)
            bpr.save(os.path.join(MODELS_DIR, "bpr_ranker.pkl"))

            # End-to-end evaluation: retrieval → BPR rerank
            log.info("\nEnd-to-end evaluation: Retrieval → BPR-MF Rerank")
            reranked = bpr.rerank_batch(retrieval_recs, top_k=20)

            # Evaluate reranked results
            e2e_scores = evaluate_retrieval(reranked, gt_adaptive, k_list=[10, 20])
            print_metrics(e2e_scores, "End-to-End (Retrieval + BPR-MF @20)")

            # Also evaluate retrieval-only at same K for comparison
            retrieval_at_20 = {uid: recs[:20] for uid, recs in retrieval_recs.items()
                               if uid in gt_adaptive}
            ret_only_scores = evaluate_retrieval(retrieval_at_20, gt_adaptive, k_list=[10, 20])
            print_metrics(ret_only_scores, "Retrieval-Only @20")

            log.info("\nEnd-to-End Comparison:")
            log.info(f"  Retrieval-Only  NDCG@10={ret_only_scores.get('NDCG@10', 0):.4f}  "
                     f"NDCG@20={ret_only_scores.get('NDCG@20', 0):.4f}")
            log.info(f"  + BPR-MF Rerank NDCG@10={e2e_scores.get('NDCG@10', 0):.4f}  "
                     f"NDCG@20={e2e_scores.get('NDCG@20', 0):.4f}")

            # Save end-to-end results
            e2e_df = pd.DataFrame({
                "Retrieval-Only": ret_only_scores,
                "Retrieval+BPR": e2e_scores,
            })
            e2e_df.to_csv(os.path.join(LOG_DIR, "week2_e2e_results.csv"))

        except Exception as e:
            log.error(f"BPR-MF ranker failed: {e}", exc_info=True)
    else:
        log.warning("No retrieval results available for BPR-MF training")

    total_time = time.time() - total_start
    log.info(f"\n{'='*60}")
    log.info(f"Week 2 training complete in {total_time/60:.1f} minutes")
    log.info(f"Log saved to {LOG_PATH}")
    log.info(f"{'='*60}")


if __name__ == "__main__":
    main()
