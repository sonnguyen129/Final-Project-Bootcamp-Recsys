"""Quick smoke-test for Week 1 preprocessing pipeline."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.data.preprocessing import (
    load_big_matrix, load_small_matrix,
    split_big_matrix, build_ground_truth, build_watch_sequences
)
import numpy as np

print("Loading big_matrix...")
big = load_big_matrix()
print(f"  shape={big.shape}  mem={big.memory_usage(deep=True).sum()/1e6:.0f} MB")
print(f"  users={big.user_id.nunique():,}  videos={big.video_id.nunique():,}")
print(f"  positive rate={big.label.mean():.4f}")

print("\nLoading small_matrix...")
small = load_small_matrix()
print(f"  shape={small.shape}")
print(f"  positive rate={small.label.mean():.4f}")

print("\nSplitting big_matrix (chronological 70/15/15)...")
train, val, test = split_big_matrix(big)

print("\nBuilding ground truth from small_matrix...")
gt = build_ground_truth(small)
print(f"  GT users with positives: {len(gt):,}")

print("\nBuilding watch sequences from train (threshold=0.5)...")
seqs = build_watch_sequences(train)
lengths = [len(s) for s in seqs.values()]
print(f"  mean seq len: {np.mean(lengths):.1f}  median: {np.median(lengths):.1f}  max: {max(lengths)}")

print("\nAll preprocessing checks PASSED.")
