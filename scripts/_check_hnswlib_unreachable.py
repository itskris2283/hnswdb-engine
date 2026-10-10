"""Check self-retrieval rate for hnswlib on the same Dataset B."""
import numpy as np
import hnswlib
import sys
sys.path.insert(0, ".")
from benchmarks.datasets_synth import compute_ground_truth_bruteforce

# Same dataset generation
rng = np.random.default_rng(42)
n_train, dim, n_clusters = 20000, 64, 50
centroids = rng.standard_normal((n_clusters, dim)).astype(np.float32)
train_labels = rng.integers(0, n_clusters, size=n_train)
noise_train = rng.standard_normal((n_train, dim)).astype(np.float32) * 0.1
xb = (centroids[train_labels] + noise_train).astype(np.float32)

# Build hnswlib index
idx = hnswlib.Index(space="l2", dim=dim)
idx.init_index(max_elements=n_train, ef_construction=100, M=16)
idx.add_items(xb, np.arange(n_train))
idx.set_ef(500)

# Self-retrieval test
n_unreachable = 0
for i in range(n_train):
    ids, dists = idx.knn_query(xb[i:i+1], k=1)
    if ids[0][0] != i:
        n_unreachable += 1

print(f"hnswlib self-retrieval: {n_train - n_unreachable}/{n_train} reachable, {n_unreachable} unreachable")
