---
name: Adaptive labeling rationale
description: Why fixed watch_ratio>2.0 is biased and how duration-adaptive thresholds work
type: project
---

Fixed threshold watch_ratio > 2.0 is heavily biased toward short videos:
- <7s videos: 12.9% positive (median wr=1.23)
- 7-15s videos: 3.3% positive (median wr=0.78)
- 15-30s videos: 0.8% positive (median wr=0.38)
- 30-60s videos: 0.2% positive (median wr=0.17)
- >60s videos: 0.1% positive (median wr=0.06)

Long videos are essentially invisible under fixed threshold. Adaptive thresholds defined in `ADAPTIVE_THRESHOLDS` dict in preprocessing.py.

**Why:** User requested exploring duration-aware labeling after seeing that most videos are 7-15s and long videos have near-zero positive rate with fixed threshold.
**How to apply:** Use `build_adaptive_ground_truth()` as primary eval. Keep `build_ground_truth()` (fixed) for comparison. When adding ranking models, consider using adaptive labels for training data too.
