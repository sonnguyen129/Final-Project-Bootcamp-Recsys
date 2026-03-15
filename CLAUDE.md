# CLAUDE.md — KuaiRec Short Video Recommender System

## Project Overview

A **two-stage retrieval-ranking recommendation system** built on the KuaiRec dataset.
Goal: portfolio project demonstrating research rigor and production readiness for big tech interviews.

- **Stage 1 — Retrieval**: Generate Top-100 candidates from 10,728 videos using embedding-based models + FAISS
- **Stage 2 — Ranking**: Rescore candidates with feature-aware and sequential models
- **Stage 3 — LLM Reranker**: Listwise reranking of Top-20 using GPT-4o-mini
- **Evaluation**: Train on `big_matrix`, evaluate on `small_matrix` (99.6% density = unbiased ground truth)

---

## Dataset

All data lives in `datasets/KuaiRec 2.0/data/` unless noted otherwise.

### File Locations

| File | Path | Description |
|------|------|-------------|
| `big_matrix.csv` | `datasets/KuaiRec 2.0/data/` | 12.5M interactions, 7,176 users × 10,728 videos — **training data** |
| `small_matrix.csv` | `datasets/KuaiRec 2.0/data/` | 4.67M interactions, 1,411 users × 3,327 videos, 99.6% density — **evaluation ground truth** |
| `item_categories.csv` | `datasets/KuaiRec 2.0/data/` | 1–4 genre tags per video (31 genres total) |
| `item_daily_features.csv` | `datasets/KuaiRec 2.0/data/` | 58 daily engagement columns per video (plays, likes, shares, etc.) |
| `user_features.csv` | `datasets/KuaiRec 2.0/data/` | Encoded user features (activity, follow counts, 18 encrypted onehot fields) |
| `social_network.csv` | `datasets/KuaiRec 2.0/data/` | Mutual friend lists for 472 users |
| `kuairec_caption_category.csv` | `datasets/KuaiRec 2.0/data/` | Chinese video captions, topic tags, 3-level category hierarchy |
| `user_features_raw.csv` | `datasets/` | Unencrypted demographics: gender, age, city tier, phone brand, ISP, app installs |
| `video_raw_categories_multi.csv` | `datasets/` | Raw multi-level category annotations with confidence scores |

### Key Fields

**Interaction matrices** (`big_matrix.csv`, `small_matrix.csv`):
- `watch_ratio` = `play_duration / video_duration` — primary engagement signal
  - `watch_ratio > 1.0` means the user rewatched
  - **Binary label**: `like = 1 if watch_ratio > 2.0` (use as positive threshold)
- `timestamp` — use for chronological train/val/test splits

**User features** (`user_features_raw.csv`):
- `gender`, `age_range`, `fre_city_level`, `phone_brand`, `mod_price`, `isp`
- `is_install_douyin` / competing app signals
- `user_active_degree` ∈ {high_active, full_active, middle_active, UNKNOWN}

**Item daily features** (`item_daily_features.csv`):
- Engagement rates: `play_cnt`, `complete_play_cnt`, `like_cnt`, `share_cnt`, `follow_cnt`
- `play_progress` = average watch ratio per day
- Aggregate over date range to get per-video features

**Item captions** (`kuairec_caption_category.csv`):
- `caption`, `manual_cover_text`, `topic_tag` — Chinese text, use for LLM reranker
- `first/second/third_level_category_name` — 3-level hierarchy

### Data Split Convention

```
big_matrix → chronological split (70% train / 15% val / 15% test)
small_matrix → held-out evaluation ONLY (never train on it)
```

No interactions overlap between the two matrices.

---

## Architecture

```
User Request
     │
     ▼
┌─────────────────────────────────────────────────────┐
│  RETRIEVAL STAGE  (FAISS index)                     │
│  ALS / Item2Vec / LightGCN / Two-Tower              │
│  Output: Top-100 candidate videos per user          │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│  RANKING STAGE                                      │
│  BPR-MF / DeepFM / GRU4Rec / SASRec                │
│  Output: Scored Top-20 candidates                   │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│  LLM RERANKER  (optional)                           │
│  GPT-4o-mini listwise reranking using captions      │
│  Output: Final Top-K recommendations                │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
             FastAPI /recommend/{user_id}
```

---

## Retrieval Models

| Model | Library | Key Params | Notes |
|-------|---------|-----------|-------|
| **ALS** | `implicit` | `confidence = 1 + 40 * watch_ratio` | Day-1 baseline, <1 min training |
| **Item2Vec** | `gensim.Word2Vec` | `sg=1, window=1000, negative=5, vector_size=128` | Sort interactions by timestamp; positive = watch_ratio > 0.5 |
| **LightGCN** | RecBole or official PyTorch repo | `n_layers=3`, BPR loss, 100+ epochs | ~16% improvement over MF; user-item bipartite graph |
| **Two-Tower** | Custom PyTorch or TFRS | In-batch negatives; user + item MLP towers | Only model using side features; handles cold-start |
| **GraphSAGE** | `torch_geometric` | `num_layers=2, hidden=128, aggregator=mean` | Inductive GNN — learns to aggregate neighbor embeddings; generalizes to unseen nodes (cold-start friendly) |

User vectors for Item2Vec: average the item embeddings of positively-watched videos.

FAISS index: start with `IndexFlatIP` (exact), then `IndexIVFFlat` (approximate) for latency benchmarks.
Target retrieval latency: **< 10ms**.

---

## Ranking Models

All four are available in **RecBole** with a unified API for direct comparison.

| Model | Paradigm | Key Config |
|-------|---------|-----------|
| **BPR-MF** | Pairwise loss baseline | `loss = -log(σ(score_pos - score_neg))` |
| **DeepFM** | Feature interaction (FM + DNN) | Feed user demographics + item categories + daily engagement stats + temporal features |
| **GRU4Rec** | Sequential RNN | Use BPR or TOP1 loss (not cross-entropy) |
| **SASRec** | Sequential Transformer | `hidden_size=64, n_heads=2, n_layers=2, max_seq_length=50`; use `SASRecF` variant for item features |

Training data for rankers: feed Top-100 retrieved candidates per user as the candidate set.
Target ranking latency: **< 50ms**.

---

## LLM Reranker

- **Model**: `gpt-4o-mini` (~$0.15/1M tokens; budget $5–20 for full eval)
- **Approach**: Listwise permutation (RankGPT-style)
- **Input**: User profile (from `user_features_raw.csv`) + last 10 watched videos with watch_ratio + Top-20 candidates with captions/categories
- **Output**: Reranked letter sequence
- **Latency**: 1–3 seconds per request (acceptable for async use)
- Chinese text in captions — GPT-4o handles it natively; translate category names for open-source models

Ablation to run: `Retrieval-only` → `Retrieval + Ranking` → `Retrieval + Ranking + LLM`

---

## Weekly Task Checklist

### Week 1 — Data Pipeline + Two Retrieval Baselines

- [ ] Load all KuaiRec files, run EDA on watch_ratio distributions and user/video statistics
- [ ] Define binary label: `like = 1 if watch_ratio > 2.0`
- [ ] Build chronological train/val/test split on big_matrix (70/15/15%)
- [ ] Create user watch sequences sorted by timestamp
- [ ] Train ALS with `implicit` library; evaluate Recall@{10,20,50} and NDCG@{10,20,50} on small_matrix
- [ ] Train Item2Vec; build user embeddings by averaging; compare with ALS
- [ ] Build FAISS index (IndexFlatIP then IndexIVFFlat); benchmark latency vs. brute-force
- [ ] Implement `user_id → Top-100 candidates` retrieval function

**Exit criteria**: Two retrieval models evaluated, FAISS pipeline functional, baseline metrics table ready.

---

### Week 2 — Advanced Retrieval + Ranking Baseline

- [ ] Implement LightGCN (RecBole or official repo); train 100+ epochs with early stopping on val NDCG
- [ ] Export LightGCN user/item embeddings; compare all 3 retrieval models
- [ ] Implement GraphSAGE (`torch_geometric`) on user-item bipartite graph with `watch_ratio` edge weights; optionally add node features from `user_features_raw.csv` and `item_daily_features.csv`; evaluate inductive cold-start performance
- [ ] Build Two-Tower model: preprocess user features (demographics + watch history aggregates) and item features (categories + daily stats); train with in-batch negatives
- [ ] Update FAISS index with best retrieval model embeddings
- [ ] Train BPR-MF ranker on Top-100 retrieved candidates; evaluate end-to-end pipeline

**Exit criteria**: Five retrieval models compared, best retrieval model selected, first end-to-end pipeline metrics.

---

### Week 3 — Advanced Ranking + Ablation Study

- [ ] Train GRU4Rec on user sequences; compare with BPR
- [ ] Implement SASRec with causal self-attention; tune sequence length (10, 25, 50, 100)
- [ ] Train DeepFM with full feature set (user demographics + item categories + engagement stats + temporal); run feature ablation
- [ ] Build full results matrix: all 4 retrieval × 4 ranking combinations
- [ ] Plot Recall@K and NDCG@K curves; run paired t-tests for statistical significance
- [ ] Generate comparison tables and figures

**Exit criteria**: Four ranking models compared, ablation study complete, best combination identified.

---

### Week 4 — LLM Reranker + API + Documentation

- [ ] Implement GPT-4o-mini listwise reranker using captions and user history
- [ ] Run 3-stage ablation: Retrieval → Retrieval+Ranking → Retrieval+Ranking+LLM
- [ ] Build FastAPI endpoint: `POST /recommend/{user_id}`
- [ ] Write Dockerfile and docker-compose.yml; add health check and Swagger docs
- [ ] Write README with architecture diagram, results tables, key findings section
- [ ] Track experiments with Weights & Biases
- [ ] Add type hints, docstrings, unit tests for data pipeline and inference
- [ ] Pin all dependency versions; final code review; push to GitHub

**Exit criteria**: End-to-end system in Docker, comprehensive docs, publishable results.

---

## Repository Structure

```
Final-Project-Bootcamp-Recsys/
├── CLAUDE.md
├── README.md
├── requirements.txt
├── docker-compose.yml
├── Dockerfile
├── datasets/
│   └── KuaiRec 2.0/data/           # raw data (do not modify)
├── notebooks/
│   ├── 01_eda.ipynb                 # EDA and preprocessing
│   ├── 02_retrieval.ipynb           # Retrieval model experiments
│   ├── 03_ranking.ipynb             # Ranking model experiments
│   └── 04_ablation.ipynb            # Full ablation study
├── src/
│   ├── data/
│   │   ├── preprocessing.py         # splits, label creation, sequences
│   │   └── features.py              # user/item feature engineering
│   ├── retrieval/
│   │   ├── als.py
│   │   ├── item2vec.py
│   │   ├── lightgcn.py
│   │   ├── graphsage.py
│   │   └── two_tower.py
│   ├── ranking/
│   │   ├── bpr.py
│   │   ├── deepfm.py
│   │   ├── gru4rec.py
│   │   └── sasrec.py
│   ├── reranker/
│   │   └── llm_reranker.py
│   ├── indexing/
│   │   └── faiss_index.py
│   └── evaluation/
│       └── metrics.py               # Recall@K, NDCG@K
├── api/
│   └── main.py                      # FastAPI app
├── models/                          # saved model artifacts (gitignored)
└── experiments/                     # W&B configs, result CSVs
```

---

## Key Conventions

- **Always train on `big_matrix`**, evaluate on `small_matrix` — never the reverse
- **Binary positive label**: `watch_ratio > 2.0` (rewatched at least twice)
- **Retrieval confidence for ALS**: `confidence = 1 + 40 * watch_ratio`
- **Item2Vec positive filter**: `watch_ratio > 0.5` when building sequences
- **Time-based splits**: chronological only — no random shuffling of interactions
- **Metrics**: Recall@{10,20,50}, NDCG@{10,20,50} — report both; NDCG is primary
- **Cold-start baseline**: compare Two-Tower vs. LightGCN on new users/items explicitly
- **Latency targets**: FAISS < 10ms, ranking inference < 50ms, LLM reranker 1–3s

---

## Tech Stack

| Component | Library/Tool |
|-----------|-------------|
| Collaborative filtering | `implicit` (ALS), RecBole (BPR, GRU4Rec, SASRec, LightGCN) |
| Item2Vec | `gensim` |
| Two-Tower / SASRec custom | `PyTorch` |
| FAISS indexing | `faiss-cpu` or `faiss-gpu` |
| Feature-based ranking | RecBole (DeepFM) |
| LLM reranker | `openai` (gpt-4o-mini) |
| API | `FastAPI`, `uvicorn` |
| Containerization | `Docker`, `docker-compose` |
| Experiment tracking | `wandb` |
| Data manipulation | `pandas`, `numpy`, `scipy` |
| Visualization | `matplotlib`, `seaborn` |

---

## Essential Papers

- **KuaiRec** — Gao et al., CIKM 2022 (dataset, unbiased evaluation)
- **YouTube DNN** — Covington et al., RecSys 2016 (two-stage architecture)
- **LightGCN** — He et al., SIGIR 2020 (simplified GCN for recommendation)
- **Item2Vec** — Barkan & Koenigstein, IEEE MLSP 2016
- **SASRec** — Kang & McAuley, ICDM 2018 (self-attentive sequential rec)
- **GRU4Rec** — Hidasi et al., ICLR 2016 (RNN sequential rec)
- **RankGPT** — EMNLP 2023 Outstanding Paper (LLM listwise reranking)
- **BARS benchmark** — Zhu et al., SIGIR 2022 (reproducibility standards)
