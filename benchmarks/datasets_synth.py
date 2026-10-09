"""
Deterministic dataset generator and loader for HNSWDB investigations and benchmarks.
Provides exact, reproducible datasets A, B, C with ground truth calculation.
"""

import os
import numpy as np
import h5py
from benchmarks.download_datasets import download_ann_benchmark_dataset, DATA_DIR


def compute_ground_truth_bruteforce(xb: np.ndarray, xq: np.ndarray, metric: str = "l2", k: int = 100):
    """Compute exact ground truth (neighbors, distances) via brute force NumPy."""
    n_q = len(xq)
    k_actual = min(k, len(xb))
    neighbors = np.empty((n_q, k_actual), dtype=np.int64)
    distances = np.empty((n_q, k_actual), dtype=np.float32)

    # Process in chunks to bound memory
    chunk_size = 500
    for start in range(0, n_q, chunk_size):
        end = min(start + chunk_size, n_q)
        q_chunk = xq[start:end]

        if metric == "l2":
            # Exact squared euclidean distance: sum((q - b)^2)
            for i, q in enumerate(q_chunk):
                diff = xb - q
                row = np.sum(diff * diff, axis=1)
                idx = np.argpartition(row, k_actual - 1)[:k_actual]
                idx = idx[np.argsort(row[idx])]
                neighbors[start + i] = idx
                distances[start + i] = row[idx]

        elif metric == "ip":
            dists = -np.dot(q_chunk, xb.T)
            for i, row in enumerate(dists):
                idx = np.argpartition(row, k_actual - 1)[:k_actual]
                idx = idx[np.argsort(row[idx])]
                neighbors[start + i] = idx
                distances[start + i] = row[idx]
        elif metric == "cosine":
            # Assume normalized vectors
            dists = 1.0 - np.dot(q_chunk, xb.T)
            for i, row in enumerate(dists):
                idx = np.argpartition(row, k_actual - 1)[:k_actual]
                idx = idx[np.argsort(row[idx])]
                neighbors[start + i] = idx
                distances[start + i] = row[idx]
        else:
            raise ValueError(f"Unknown metric: {metric}")

    return neighbors, distances


def get_dataset_a(seed: int = 42):
    """Dataset A: 5,000 vectors, 32 dimensions, 20 clusters, 1,000 test queries."""
    rng = np.random.default_rng(seed)
    n_train, n_test, dim, n_clusters = 5000, 1000, 32, 20
    centroids = rng.standard_normal((n_clusters, dim)).astype(np.float32)
    train_labels = rng.integers(0, n_clusters, size=n_train)
    noise_train = rng.standard_normal((n_train, dim)).astype(np.float32) * 0.1
    xb = (centroids[train_labels] + noise_train).astype(np.float32)

    test_labels = rng.integers(0, n_clusters, size=n_test)
    noise_test = rng.standard_normal((n_test, dim)).astype(np.float32) * 0.1
    xq = (centroids[test_labels] + noise_test).astype(np.float32)

    gt_nbrs, gt_dists = compute_ground_truth_bruteforce(xb, xq, metric="l2", k=100)
    return xb, xq, gt_nbrs, gt_dists, "l2"


def get_dataset_b(seed: int = 42):
    """Dataset B: 20,000 vectors, 64 dimensions, 50 clusters, 1,000 test queries."""
    rng = np.random.default_rng(seed)
    n_train, n_test, dim, n_clusters = 20000, 1000, 64, 50
    centroids = rng.standard_normal((n_clusters, dim)).astype(np.float32)
    train_labels = rng.integers(0, n_clusters, size=n_train)
    noise_train = rng.standard_normal((n_train, dim)).astype(np.float32) * 0.1
    xb = (centroids[train_labels] + noise_train).astype(np.float32)

    test_labels = rng.integers(0, n_clusters, size=n_test)
    noise_test = rng.standard_normal((n_test, dim)).astype(np.float32) * 0.1
    xq = (centroids[test_labels] + noise_test).astype(np.float32)

    gt_nbrs, gt_dists = compute_ground_truth_bruteforce(xb, xq, metric="l2", k=100)
    return xb, xq, gt_nbrs, gt_dists, "l2"


def get_dataset_c():
    """Dataset C: Fashion-MNIST (60,000 train, 10,000 test, 784 dimensions)."""
    path = os.path.join(DATA_DIR, "fashion-mnist-784-euclidean.hdf5")
    if not os.path.exists(path):
        download_ann_benchmark_dataset("fashion-mnist-784-euclidean")
    with h5py.File(path, "r") as f:
        xb = np.array(f["train"], dtype=np.float32)
        xq = np.array(f["test"], dtype=np.float32)
        gt_nbrs = np.array(f["neighbors"]) if "neighbors" in f else None
        gt_dists = np.array(f["distances"], dtype=np.float32) if "distances" in f else None
    return xb, xq, gt_nbrs, gt_dists, "l2"
