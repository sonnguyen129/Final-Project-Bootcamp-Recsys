---
name: KuaiRec evaluation protocol
description: How to correctly evaluate retrieval models on KuaiRec — FAISS pool filtering, adaptive labels, recall formula
type: feedback
---

Models should ONLY produce embeddings. FAISS handles all retrieval inference. Do not add model-specific recommend_from_pool() methods.

Evaluation must restrict FAISS index to small_matrix videos only (3,327 items). big_matrix eval users interact with completely disjoint video sets from small_matrix — recommending from full 10k+ catalog yields zero overlap with ground truth.

Use adaptive ground truth (duration-based thresholds) as primary metric. Also report fixed (>2.0) for comparison.

Recall@K denominator = len(relevant), NOT min(len(relevant), k). The latter makes Recall flat as K increases.

**Why:** User corrected multiple issues: zero metrics from wrong candidate pool, flat Recall from wrong formula, pool-based eval methods that bypass FAISS.
**How to apply:** For any new retrieval model, use `evaluate_via_faiss()` from 02_retrieval.ipynb. Never evaluate by having models recommend directly.
