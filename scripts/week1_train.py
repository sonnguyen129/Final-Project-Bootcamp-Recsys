"""
Week 1 training script.
Trains ALS and Item2Vec, evaluates on small_matrix, builds FAISS index,
benchmarks latency, saves results to experiments/week1_results.csv.
"""
import os, sys, json, time
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# ---------- Tee stdout/stderr to log file ----------
import io
_LOG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                         "experiments", "week1_train.log")
os.makedirs(os.path.dirname(_LOG_PATH), exist_ok=True)

class _Tee(io.TextIOWrapper):
    def __init__(self, stream, log_file):
        self._stream = stream
        self._log    = open(log_file, "w", encoding="utf-8")
    def write(self, data):
        self._stream.write(data)
        self._log.write(data)
        self._log.flush()
        return len(data)
    def flush(self):
        self._stream.flush()
        self._log.flush()

sys.stdout = _Tee(sys.stdout, _LOG_PATH)
sys.stderr = _Tee(sys.stderr, _LOG_PATH)
# ---------------------------------------------------

import numpy as np
import pandas as pd
from pathlib import Path

from src.data.preprocessing import (
    load_big_matrix, load_small_matrix,
    split_big_matrix, build_ground_truth
)
from src.retrieval.als import ALSRetriever
from src.retrieval.item2vec import Item2VecRetriever
from src.indexing.faiss_index import FAISSIndex
from src.evaluation.metrics import evaluate_retrieval, print_metrics

EXP_DIR   = Path("experiments")
MODEL_DIR = Path("models")
EXP_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

# =========================================================================
# 1. Load data
# =========================================================================
print("\n[1/7] Loading data...")
big   = load_big_matrix()
small = load_small_matrix()

print(f"big_matrix:   {big.shape}   users={big.user_id.nunique():,}  videos={big.video_id.nunique():,}")
print(f"small_matrix: {small.shape}   users={small.user_id.nunique():,}  videos={small.video_id.nunique():,}")

# =========================================================================
# 2. Split
# =========================================================================
print("\n[2/7] Chronological split (70/15/15)...")
train, val, test = split_big_matrix(big)
del val, test   # free memory - not needed for retrieval eval

# =========================================================================
# 3. Ground truth
# =========================================================================
print("\n[3/7] Building ground truth from small_matrix...")
ground_truth = build_ground_truth(small)
del small

train_users = set(train["user_id"].unique())
eval_users  = [u for u in ground_truth if u in train_users]
print(f"Eval users (in train & small_matrix): {len(eval_users):,}")

# =========================================================================
# 4. ALS
# =========================================================================
print("\n[4/7] Training ALS...")
t0 = time.time()
als = ALSRetriever(factors=128, iterations=30, alpha=40)
als.fit(train)
print(f"ALS trained in {time.time()-t0:.1f}s")
als.save(str(MODEL_DIR / "als.pkl"))

print("Evaluating ALS on small_matrix...")
als_recs   = als.recommend_batch(eval_users, n=50)
als_scores = evaluate_retrieval(als_recs, ground_truth, k_list=[10, 20, 50])
print_metrics(als_scores, "ALS")

# =========================================================================
# 5. Item2Vec
# =========================================================================
print("\n[5/7] Training Item2Vec...")
t0 = time.time()
i2v = Item2VecRetriever(vector_size=128, window=1000, epochs=10)
i2v.fit(train)
print(f"Item2Vec trained in {time.time()-t0:.1f}s")
i2v.save(str(MODEL_DIR / "item2vec.pkl"))

print("Evaluating Item2Vec on small_matrix...")
i2v_eval  = [u for u in eval_users if u in i2v.user_embeddings]
i2v_recs  = i2v.recommend_batch(i2v_eval, n=50)
i2v_scores = evaluate_retrieval(i2v_recs, ground_truth, k_list=[10, 20, 50])
print_metrics(i2v_scores, "Item2Vec")

# =========================================================================
# 6. FAISS index (on ALS embeddings)
# =========================================================================
print("\n[6/7] Building FAISS index (ALS embeddings)...")
item_ids  = list(als.idx2item.values())
item_embs = als.get_item_embeddings()
user_embs = als.get_user_embeddings()

idx_exact = FAISSIndex(mode="exact")
idx_exact.build(item_ids, item_embs)
bench_exact = idx_exact.benchmark(user_embs, k=100, n_queries=200)
idx_exact.save(str(MODEL_DIR / "faiss_als_exact.index"),
               str(MODEL_DIR / "faiss_als_exact_meta.json"))

idx_approx = FAISSIndex(mode="approx", n_list=100)
idx_approx.build(item_ids, item_embs)
bench_approx = idx_approx.benchmark(user_embs, k=100, n_queries=200)
idx_approx.save(str(MODEL_DIR / "faiss_als_approx.index"),
                str(MODEL_DIR / "faiss_als_approx_meta.json"))

# Quick eval: FAISS-backed ALS
def faiss_recommend_batch(faiss_idx, uids, user2idx, user_embs_arr, n=50):
    recs = {}
    for uid in uids:
        if uid not in user2idx:
            continue
        vec = user_embs_arr[user2idx[uid]]
        recs[uid] = faiss_idx.search(vec, k=n)
    return recs

faiss_recs   = faiss_recommend_batch(idx_exact, eval_users, als.user2idx, user_embs)
faiss_scores = evaluate_retrieval(faiss_recs, ground_truth, k_list=[10, 20, 50])
print_metrics(faiss_scores, "ALS + FAISS (exact)")

# =========================================================================
# 7. Save results
# =========================================================================
print("\n[7/7] Saving results...")
rows = [
    {"Model": "ALS",              **als_scores},
    {"Model": "Item2Vec",         **i2v_scores},
    {"Model": "ALS+FAISS(exact)", **faiss_scores},
]
results_df = pd.DataFrame(rows).set_index("Model")
results_df = results_df[sorted(results_df.columns)].round(4)
out_path = EXP_DIR / "week1_results.csv"
results_df.to_csv(out_path)
print(f"\nResults saved to {out_path}")
print("\n" + results_df.to_string())

latency = {
    "faiss_exact":  bench_exact,
    "faiss_approx": bench_approx,
}
with open(EXP_DIR / "week1_faiss_latency.json", "w") as f:
    json.dump(latency, f, indent=2)
print(f"\nFAISS benchmark saved to experiments/week1_faiss_latency.json")
print(f"\nExact  index: mean={bench_exact['latency_ms_mean']:.2f}ms  p95={bench_exact['latency_ms_p95']:.2f}ms")
print(f"Approx index: mean={bench_approx['latency_ms_mean']:.2f}ms  p95={bench_approx['latency_ms_p95']:.2f}ms")
print("\nWeek 1 training complete.")
