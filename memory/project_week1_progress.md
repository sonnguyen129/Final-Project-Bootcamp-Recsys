---
name: Week 1 progress
description: Week 1 implementation status — data pipeline, ALS, Item2Vec, FAISS, adaptive labeling
type: project
---

## Week 1 — Data Pipeline + Two Retrieval Baselines

### Completed
- EDA on watch_ratio distributions, video duration, user/video stats (01_eda.ipynb)
- Adaptive labeling: duration-based thresholds instead of fixed watch_ratio > 2.0
  - <7s: wr>2.0 | 7-15s: wr>1.0 | 15-30s: wr>0.5 | 30-60s: wr>0.3 | >60s: wr>0.2
  - `build_adaptive_ground_truth()` in preprocessing.py
- Chronological train/val/test split on big_matrix (70/15/15%)
- ALS trained (implicit library, 128 factors, 30 iterations, alpha=40)
- Item2Vec trained (gensim Word2Vec, sg=1, window=1000, 128 dims, 10 epochs)
- FAISS index (exact + approx) built and benchmarked
- Shared evaluation pipeline: model embeddings → FAISS (small_matrix items only) → Recall/NDCG@{10,20,50,100}
- Recall@K formula fixed: denominator = len(relevant), not min(len(relevant), k)
- User watch sequences validated (7,174 users, mean len=792, used by Item2Vec successfully)
- `user_id → Top-100 candidates` retrieval function implemented via `evaluate_via_faiss()` in 02_retrieval.ipynb

### Baseline results (adaptive ground truth)
| Model    | NDCG@10 | NDCG@100 | Recall@10 | Recall@100 |
|----------|---------|----------|-----------|------------|
| ALS      | 0.2029  | 0.2128   | 0.0025    | 0.0262     |
| Item2Vec | 0.2947  | 0.2457   | 0.0035    | 0.0301     |

FAISS latency: exact=0.36ms, approx=0.21ms (target <10ms ✓)

### Key design decisions
- Eval protocol: restrict FAISS candidate pool to 3,327 small_matrix videos only (KuaiRec unbiased eval)
- big_matrix eval users have disjoint video sets from small_matrix — must filter FAISS index to small_matrix items
- Models only produce embeddings; FAISS is the shared inference layer for all models
- Both adaptive and fixed (>2.0) ground truth reported for comparison
- Item2Vec window reduced to 50 (from 1000) for practical training time

### Week 1 status: COMPLETE

**Why:** Portfolio project for big tech interviews, demonstrating research rigor and production readiness.
**How to apply:** When adding new retrieval models (LightGCN, Two-Tower, GraphSAGE), follow the same pattern: train → extract embeddings → evaluate_via_faiss().
