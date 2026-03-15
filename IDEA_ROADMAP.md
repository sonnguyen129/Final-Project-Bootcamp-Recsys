# Building a short video recommender that lands big tech interviews

**A two-stage retrieval-ranking system built on KuaiRec — the only fully-observed real-world recommendation dataset — is among the strongest portfolio projects a data scientist can present to big tech reviewers.** This roadmap covers the complete pipeline: data preprocessing, 4 retrieval models, 4 ranking models, an LLM reranker, FAISS indexing, and a Dockerized API — all achievable in 4 weeks. The KuaiRec dataset's unique 99.6% density enables unbiased offline evaluation impossible with any other public dataset, giving your project a built-in differentiator. What follows is a deeply researched, week-by-week plan with model recommendations, implementation specifics, and portfolio optimization strategies.

---

## KuaiRec is unlike any other recommendation dataset

KuaiRec, collected from Kuaishou (China's second-largest short video platform), is the **first real-world dataset with a near-fully-observed user-item interaction matrix** at 99.6% density. Published at CIKM 2022, it was constructed by deliberately inserting unwatched videos into user feeds over 15 days, ensuring almost every user-item pair has recorded feedback. This eliminates the Missing-Not-At-Random (MNAR) bias that plagues every other recommendation dataset.

The dataset ships in two matrices. The **big matrix** contains **12.5 million interactions** across 7,176 users and 10,728 videos — this is your training data. The **small matrix** contains **4.67 million interactions** across 1,411 users and 3,327 videos at 99.6% density — this is your ground-truth evaluation set. Critically, no interactions overlap between the two matrices.

**Core interaction signal** is `watch_ratio` (play_duration / video_duration), a continuous value where >1.0 means the user rewatched. There is no per-interaction like/follow signal — you must threshold watch_ratio to create binary labels (e.g., positive if watch_ratio > 2.0). Both matrices include full timestamps (July 5 – September 5, 2020), enabling proper time-based splits.

The supporting files are rich. **`user_features_raw.csv`** provides unencrypted demographics for all 7,176 users: gender, age range, city tier, phone brand/model/price, province, ISP, and whether they have competing apps (Douyin/TikTok) installed. **`item_daily_features.csv`** contains 58 columns of daily engagement statistics per video — impressions, play counts, completion rates, likes, comments, shares, downloads, follows, and reports — spanning the full collection period. **`item_categories.csv`** assigns each video 1–4 tags from 31 genres. **`kuairec_caption_category.csv`** provides Chinese-language video captions, cover text, topic tags, and a three-level category hierarchy. A **social network file** covers 472 users with mutual friend lists.

For your project, the critical design decision is: train on big_matrix, evaluate on small_matrix. The small matrix's full observability means your Recall@K and NDCG@K scores reflect true performance, not artifacts of exposure bias.

---

## Four retrieval models, ordered from baseline to state-of-the-art

The retrieval stage generates Top-N candidates from the full item catalog. Implement these four models in order — each introduces a new paradigm and builds complexity progressively.

**ALS (Alternating Least Squares)** is your day-one baseline. It decomposes the user-item matrix R ≈ U × V^T using the implicit feedback formulation where watch_ratio maps directly to confidence weighting. The `implicit` library trains this in under 60 seconds on KuaiRec's scale, producing both user and item embeddings. ALS captures global collaborative patterns but cannot leverage side features or sequential information. Implementation is roughly 10 lines of code. Use `confidence = 1 + 40 * watch_ratio` as your starting point.

**Item2Vec** adapts Word2Vec's skip-gram architecture to items. Sort each user's interactions by timestamp to create "sentences" of video IDs, then train with Gensim's `Word2Vec(sg=1, window=1000, negative=5)`. This captures co-viewing patterns — videos frequently watched by the same users cluster together in embedding space. The limitation is that Item2Vec produces only item embeddings; user representations require averaging their watched items' vectors. Train with `vector_size=128` and filter sequences to only include positive interactions (watch_ratio > 0.5).

**LightGCN** represents the graph neural network paradigm and is the current state-of-the-art for pure collaborative filtering. It constructs a user-item bipartite graph, then propagates embeddings through 2–3 layers of simplified graph convolution — just neighborhood averaging with symmetric normalization, no feature transforms or activations. The key insight from He et al. (SIGIR 2020) is that removing these components actually *improves* recommendation performance. Train with BPR loss on positive edges (watch_ratio > threshold). Use the official LightGCN-PyTorch repository or RecBole's implementation. Expect **~16% improvement over matrix factorization** baselines.

**Two-Tower Model** is the most complex but highest-potential retrieval approach, and the industry standard at YouTube, TikTok, and Instagram. Two separate MLPs encode user features and item features independently into a shared embedding space, with affinity measured by dot product. The user tower ingests demographics from `user_features_raw.csv` plus aggregated watch history; the item tower ingests categories, duration, engagement stats from `item_daily_features.csv`, and optionally text features. This is the only retrieval model that handles cold-start and leverages KuaiRec's rich feature set. Use TensorFlow Recommenders (TFRS) or custom PyTorch with in-batch negative sampling.

| Model | Complexity | Training time | Uses features | Cold-start | Expected quality |
|-------|-----------|--------------|---------------|------------|-----------------|
| ALS | Very low | <1 min | No | No | Good baseline |
| Item2Vec | Low | ~5 min | No | No | Good (item-only) |
| LightGCN | Medium | 10–30 min (GPU) | No | No | High |
| Two-Tower | High | 30 min–2 hr (GPU) | Yes | Yes | Highest potential |

---

## Four ranking models that showcase ML research depth

The ranking stage rescores the Top-N retrieved candidates using richer signals. These four models span distinct paradigms — pairwise loss, feature interaction, RNN sequential, and Transformer sequential — creating a compelling ablation narrative.

**BPR-MF (Bayesian Personalized Ranking)** serves as the ranking baseline. It optimizes a pairwise loss: for each user, the model learns to score observed items higher than unobserved ones via `loss = -log(σ(score_pos - score_neg))`. This directly optimizes for ranking rather than pointwise prediction. BPR is fast, interpretable, and available in RecBole with a single function call. It establishes the floor that more complex models must beat.

**DeepFM** combines Factorization Machines (automatic pairwise feature interactions) with a deep neural network (higher-order interactions) in a parallel architecture sharing the same embedding layer. This is KuaiRec's feature-rich model: feed it user demographics, video categories, engagement statistics, video duration, and temporal features. DeepFM eliminates the manual cross-feature engineering required by Wide & Deep. Use it to demonstrate whether KuaiRec's rich side information improves ranking beyond pure collaborative signals.

**GRU4Rec** is the foundational sequential model, applying GRU cells to user watch sequences to predict the next item. It captures temporal dynamics in short video consumption — the swipe-based browsing pattern maps naturally to sequential modeling. Use ranking-based loss (BPR or TOP1) rather than cross-entropy. GRU4Rec serves as the sequential baseline that SASRec must outperform.

**SASRec (Self-Attentive Sequential Recommendation)** is the primary sequential model and likely your strongest ranker. It replaces GRU's recurrence with Transformer self-attention using causal masking, enabling parallel training and better long-range dependency capture. On dense datasets like KuaiRec, SASRec adaptively attends to longer histories. The original paper reports **6.9% Hit Rate and 9.6% NDCG gains** over the strongest baselines. Typical architecture: `hidden_size=64, n_heads=2, n_layers=2, max_seq_length=50`. RecBole also offers `SASRecF` which incorporates item features.

All four models are available in **RecBole** with a unified API, enabling direct apples-to-apples comparison. The progression BPR → GRU4Rec → SASRec tells the story of collaborative filtering → sequential RNN → sequential Transformer. DeepFM complements this by exploring the feature interaction paradigm.

---

## The LLM reranker adds a cutting-edge differentiator

After your best ranking model produces a scored Top-20 list, an LLM reranker rescores these candidates using natural language understanding of video content and user preferences. This is a rapidly evolving research area — the RankGPT paper (EMNLP 2023, Outstanding Paper Award) demonstrated that GPT-4 can perform competitive listwise reranking via instructional permutation generation.

**Prompt design for KuaiRec** should follow a listwise approach. Construct prompts using data from `kuairec_caption_category.csv` (video captions, topic tags, hierarchical categories) and the user's recent watch history with watch ratios. Since KuaiRec's text is in Chinese, either use a multilingual model (GPT-4o handles Chinese well) or translate category names to English for open-source models.

```
System: You are a short video recommendation expert. Rerank candidates 
by predicted user engagement.

User Profile: Female, 24-30, high_active, prefers [top 3 categories]

Recent Watch History (last 10 videos):
1. "宠物狗日常" (Pet dog daily) - Pets - Watch ratio: 3.2
2. "搞笑合集" (Funny compilation) - Comedy - Watch ratio: 2.8
...

Candidates to Rerank:
A. "猫咪卖萌" (Cute cat) - Pets - 15s - High engagement
B. "美食教程" (Food tutorial) - Cooking - 30s - Medium engagement
...

Output the reranked order as: [letter sequence]
```

Use **GPT-4o-mini** for the portfolio project — it balances cost ($0.15/1M input tokens) with strong instruction-following. Budget approximately **$5–20** for full evaluation across KuaiRec's 1,411 test users. The real value is the ablation: compare Retrieval-only vs. Retrieval+Ranking vs. Retrieval+Ranking+LLM to quantify each stage's marginal contribution. Research findings on LLM reranking improvements are mixed, which makes honest analysis even more impressive to reviewers.

---

## Week-by-week roadmap for a 4-week build

### Week 1: Data pipeline + two retrieval baselines

| Day | Task | Deliverable |
|-----|------|-------------|
| 1–2 | **EDA and preprocessing.** Load all KuaiRec files. Explore watch_ratio distributions, user activity patterns, video popularity curves. Define binary label threshold (watch_ratio > 2.0 for positive). Build time-based train/val/test split on big_matrix (70/15/15% chronological). Create user watch sequences sorted by timestamp. | EDA notebook, preprocessed data files, train/val/test splits |
| 3 | **ALS baseline.** Build sparse user-item matrix from big_matrix. Train ALS with `implicit` library. Extract user and item embeddings. Evaluate Recall@{10,20,50}, NDCG@{10,20,50} on small_matrix ground truth. | ALS model, embeddings, baseline metrics |
| 4–5 | **Item2Vec.** Generate user watch sequences (positive items only, chronological). Train Word2Vec skip-gram. Build user representations by averaging item embeddings. Evaluate against ALS. | Item2Vec embeddings, comparison table |
| 6–7 | **FAISS index.** Build FAISS index from best retrieval embeddings. Implement exact search (IndexFlatIP) and approximate search (IndexIVFFlat). Benchmark retrieval latency and recall vs. brute-force. Write retrieval serving function: `user_id → Top-100 candidates`. | FAISS index, retrieval pipeline, latency benchmarks |

**Week 1 exit criteria:** Two retrieval models evaluated, FAISS pipeline functional, baseline metrics established.

### Week 2: Advanced retrieval + ranking baseline

| Day | Task | Deliverable |
|-----|------|-------------|
| 1–3 | **LightGCN.** Construct user-item bipartite graph. Implement 3-layer LightGCN with BPR loss in PyTorch (use RecBole or official repo). Train for 100+ epochs with early stopping on validation NDCG. Export final user/item embeddings. Compare against ALS and Item2Vec. | LightGCN model, embeddings, 3-model comparison |
| 4–5 | **Two-Tower model.** Preprocess user features (demographics, activity stats, watch history aggregation) and item features (categories, duration, daily engagement stats). Build user and item tower MLPs. Train with in-batch negatives. Export embeddings and update FAISS index. | Two-Tower model, feature-enriched embeddings |
| 6–7 | **BPR ranking baseline.** Using best retrieval model's Top-100 candidates per user, construct ranking training data (positive/negative pairs from candidate set). Train BPR-MF ranker. Evaluate full pipeline: retrieval → ranking. | BPR ranker, first end-to-end pipeline metrics |

**Week 2 exit criteria:** Four retrieval models compared, best retrieval model selected, ranking pipeline established with BPR baseline.

### Week 3: Advanced ranking models + ablation

| Day | Task | Deliverable |
|-----|------|-------------|
| 1–2 | **GRU4Rec.** Train on user sequences with ranking loss. Evaluate as ranker on retrieved candidates. Compare with BPR. | GRU4Rec model, sequential vs. non-sequential comparison |
| 3–4 | **SASRec.** Implement with causal self-attention. Tune key hyperparameters (sequence length, heads, layers). This should be your strongest model. Conduct sequence length ablation (10, 25, 50, 100). | SASRec model, ablation results |
| 5–6 | **DeepFM.** Engineer features from all KuaiRec files: user demographics, item categories, video duration, daily engagement stats, temporal features. Train DeepFM. Conduct feature ablation (with/without each feature group). | DeepFM model, feature importance analysis |
| 7 | **Comprehensive ablation study.** Build results matrix: all retrieval × ranking model combinations. Plot Recall@K and NDCG@K curves. Run paired t-tests for statistical significance. Generate publication-quality tables and figures. | Full ablation table, significance tests, plots |

**Week 3 exit criteria:** Four ranking models compared, ablation study complete, best retrieval+ranking combination identified.

### Week 4: LLM reranker + API + documentation

| Day | Task | Deliverable |
|-----|------|-------------|
| 1–2 | **LLM reranker.** Implement GPT-4o-mini listwise reranking using video captions and user history. Run on test set. Complete 3-stage ablation: Retrieval → Retrieval+Ranking → Retrieval+Ranking+LLM. Analyze when LLM helps vs. hurts. | LLM reranker, 3-stage ablation results |
| 3–4 | **FastAPI + Docker.** Build FastAPI endpoint: `POST /recommend/{user_id}` → retrieval (FAISS) → ranking (SASRec) → optional LLM reranking → Top-K response. Write Dockerfile and docker-compose.yml. Add health check and Swagger docs. | Working API, Docker deployment |
| 5–6 | **Documentation and visualization.** Write comprehensive README with architecture diagram, results tables, and reproduction instructions. Create experiment tracking summary (use W&B or MLflow screenshots). Write 1-page technical blog post explaining key findings. | README, architecture diagram, blog post |
| 7 | **Polish.** Add type hints, docstrings, unit tests for data pipeline and model inference. Pin all dependency versions. Final code review and cleanup. Push to GitHub. | Production-quality repository |

**Week 4 exit criteria:** Complete end-to-end system deployed in Docker, comprehensive documentation, publishable results.

---

## Making this project impossible to ignore on a resume

**The single most impactful differentiator** is demonstrating rigorous experimental methodology — not just running models, but systematically analyzing what works and why. Big tech interviewers consistently report that ablation studies, error analysis, and honest discussion of limitations signal research maturity far more than model complexity.

Structure your GitHub repository with clear separation between exploration (`notebooks/`) and production code (`src/`). The README should open with an architecture diagram showing the two-stage pipeline, followed by a results table comparing all model combinations with bold highlighting on the best scores and confidence intervals. Include a "Key Findings" section with 3–5 non-obvious insights (e.g., "LightGCN outperformed Two-Tower on warm users but underperformed on cold-start scenarios" or "SASRec's gains over GRU4Rec were statistically significant only for users with 50+ interactions").

Leverage KuaiRec's unique properties explicitly. Mention that you evaluated on a **fully-observed ground truth matrix** — no other portfolio project can claim unbiased offline evaluation. Discuss exposure bias and how traditional evaluation on sparse test sets inflates metrics for popular items. If time permits, implement **offline A/B test simulation** using the small matrix's full observability to compare different recommendation policies — this directly mirrors production evaluation at companies like Meta and Google.

Write custom implementations for **2 key models** (e.g., Item2Vec and SASRec from scratch in PyTorch) while using RecBole for baselines. This hybrid approach shows you understand the algorithms deeply while also knowing how to leverage existing tools efficiently. Log experiments with **Weights & Biases** — the shareable experiment dashboard alone can replace pages of results documentation.

The Docker + FastAPI deployment transforms your project from an academic exercise into a production-ready system. Include realistic latency benchmarks: retrieval via FAISS should complete in <10ms, ranking inference in <50ms, with the LLM reranker adding 1–3 seconds. Discussing these latency tradeoffs in your README demonstrates the production thinking that separates ML engineers from researchers.

---

## Essential papers to cite and study

The foundational references for this project span dataset, architecture, and model families. The **KuaiRec paper** (Gao et al., CIKM 2022) introduced the fully-observed dataset and analyzed how density affects evaluation — read it first. The **YouTube DNN paper** (Covington et al., RecSys 2016) established the two-stage retrieval-ranking architecture that your entire project mirrors. For retrieval models, study **Item2Vec** (Barkan & Koenigstein, IEEE MLSP 2016) for skip-gram item embeddings and **LightGCN** (He et al., SIGIR 2020) for the insight that simpler GCNs outperform complex ones. For ranking, **GRU4Rec** (Hidasi et al., ICLR 2016) pioneered RNN-based sequential recommendation, while **SASRec** (Kang & McAuley, ICDM 2018) demonstrated Transformer superiority for the same task. **ByteDance's Monolith** (Liu et al., RecSys Workshop 2022) provides a window into TikTok's production recommendation system with real-time training. Finally, the **BARS benchmark** (Zhu et al., SIGIR 2022) establishes the reproducibility standards your project should follow — 8,000+ experiments across 70+ models.

---

## Conclusion

This project works as a portfolio piece because it mirrors real production systems while demonstrating research rigor. The two-stage architecture directly reflects how YouTube, TikTok, and Instagram serve recommendations. The 4×4 model grid (four retrieval, four ranking models) produces a rich ablation narrative that reveals not just what works, but *why*. KuaiRec's fully-observed matrix gives your evaluation credibility that no MovieLens or Amazon Reviews project can match. The LLM reranker adds a timely, cutting-edge component that signals awareness of the field's frontier. And the Docker-served API proves you can ship, not just experiment. Prioritize depth of analysis over breadth of models — a well-analyzed comparison of 3 models outweighs a shallow survey of 10. The goal is not to build the best recommendation system in the world, but to demonstrate that you *think* like someone who could.