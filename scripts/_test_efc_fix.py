"""Test: Does using simple closest-N reselection instead of heuristic fix the gap?"""
import numpy as np
import sys
sys.path.insert(0, ".")

import hnswdb
from benchmarks.datasets_synth import compute_ground_truth_bruteforce
from benchmarks.metrics import recall_at_k

# Generate Dataset B sigma=0.1
rng = np.random.default_rng(42)
n_train, dim, n_clusters = 20000, 64, 50
centroids = rng.standard_normal((n_clusters, dim)).astype(np.float32)
train_labels = rng.integers(0, n_clusters, size=n_train)
noise_train = rng.standard_normal((n_train, dim)).astype(np.float32) * 0.1
xb = (centroids[train_labels] + noise_train).astype(np.float32)

test_labels = rng.integers(0, n_clusters, size=1000)
noise_test = rng.standard_normal((1000, dim)).astype(np.float32) * 0.1
xq = (centroids[test_labels] + noise_test).astype(np.float32)

gt_nbrs, _ = compute_ground_truth_bruteforce(xb, xq, "l2", 10)

# Test current behavior
idx = hnswdb.HNSWIndex(dim=64, metric="l2", max_elements=20000, M=16, ef_construction=100, seed=42)
idx.add(xb, np.arange(20000, dtype=np.int64), num_threads=1)

ids, _ = idx.search(xq, k=10, ef_search=200)
rec = recall_at_k(ids, gt_nbrs, 10)
print(f"Current (efC=100, seed=42):  recall@10 = {rec:.4f}")

# Test with higher efC (simulates better construction)
idx2 = hnswdb.HNSWIndex(dim=64, metric="l2", max_elements=20000, M=16, ef_construction=400, seed=42)
idx2.add(xb, np.arange(20000, dtype=np.int64), num_threads=1)

ids2, _ = idx2.search(xq, k=10, ef_search=200)
rec2 = recall_at_k(ids2, gt_nbrs, 10)
print(f"Higher efC (efC=400, seed=42): recall@10 = {rec2:.4f}")

# Count unreachable with efC=400
n_unreachable = 0
for i in range(n_train):
    v = np.ascontiguousarray(xb[i:i+1], dtype=np.float32)
    ids_sr, _ = idx2.search(v, k=1, ef_search=500)
    if ids_sr[0][0] != i:
        n_unreachable += 1
print(f"Unreachable nodes (efC=400): {n_unreachable}")

diag = idx.diagnostics()
diag2 = idx2.diagnostics()
print(f"\nDiagnostics comparison:")
print(f"  efC=100: overflow_reselections={diag.get('cumulative_overflow_reselections',0):.0f}, heuristic_calls={diag.get('cumulative_heuristic_calls',0):.0f}")
print(f"  efC=400: overflow_reselections={diag2.get('cumulative_overflow_reselections',0):.0f}, heuristic_calls={diag2.get('cumulative_heuristic_calls',0):.0f}")
print(f"  efC=100: reciprocal={diag.get('reciprocal_links_fraction',0):.4f}")
print(f"  efC=400: reciprocal={diag2.get('reciprocal_links_fraction',0):.4f}")
