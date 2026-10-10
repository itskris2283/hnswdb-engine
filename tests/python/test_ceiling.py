"""
Tests for recall convergence, monotonicity, and construction quality.
Tests cover both easy and hard (tight cluster) data distributions.
"""

import numpy as np
import pytest
from hnswdb import HNSWIndex
from benchmarks.datasets_synth import get_dataset_a, get_dataset_b, compute_ground_truth_bruteforce
from benchmarks.metrics import recall_at_k, self_retrieval_rate


def test_recall_monotonicity_dataset_a():
    """Verify that recall is non-decreasing with ef_search on Dataset A."""
    xb, xq, gt_nbrs, _, _ = get_dataset_a()
    n_train = len(xb)
    dim = xb.shape[1]

    idx = HNSWIndex(dim=dim, metric="l2", max_elements=n_train, M=16, ef_construction=100, seed=42)
    idx.add(xb, np.arange(n_train, dtype=np.int64))

    recalls = []
    for ef in [10, 20, 50, 100, 200]:
        idx.set_ef_search(ef)
        ids, _ = idx.search(xq, k=10)
        rec = recall_at_k(ids, gt_nbrs, k=10)
        recalls.append(rec)

    # Monotonicity check
    for i in range(len(recalls) - 1):
        assert recalls[i+1] >= recalls[i] - 1e-4, (
            f"Recall dropped from ef={[10,20,50,100,200][i]}: {recalls[i]:.4f} "
            f"to ef={[10,20,50,100,200][i+1]}: {recalls[i+1]:.4f}"
        )

    # Dataset A (5K x 32d, 20 clusters, sigma=0.1) should achieve high recall (>= 0.990)
    assert recalls[-1] >= 0.990, f"Dataset A recall at ef=200: {recalls[-1]:.4f} < 0.990"


def test_recall_convergence_dataset_b():
    """Verify that Dataset B recall converges with increasing efC.

    KNOWN LIMITATION: At efC=100, HNSWDB achieves ~0.97 on tight clusters (sigma=0.1)
    while hnswlib achieves ~1.0 on the same data. At efC=400, HNSWDB also reaches ~1.0.
    This test documents the limitation.
    """
    rng = np.random.default_rng(42)
    n_train, n_test, dim, n_clusters = 20000, 500, 64, 50
    centroids = rng.standard_normal((n_clusters, dim)).astype(np.float32)
    train_labels = rng.integers(0, n_clusters, size=n_train)
    noise_train = rng.standard_normal((n_train, dim)).astype(np.float32) * 0.1
    xb = (centroids[train_labels] + noise_train).astype(np.float32)
    test_labels = rng.integers(0, n_clusters, size=n_test)
    noise_test = rng.standard_normal((n_test, dim)).astype(np.float32) * 0.1
    xq = (centroids[test_labels] + noise_test).astype(np.float32)
    gt_nbrs, _ = compute_ground_truth_bruteforce(xb, xq, "l2", 10)

    # Test that recall improves with efC
    recalls_by_efc = {}
    for efc in [100, 400]:
        idx = HNSWIndex(dim=dim, metric="l2", max_elements=n_train, M=16, ef_construction=efc, seed=42)
        idx.add(xb, np.arange(n_train, dtype=np.int64))
        ids, _ = idx.search(xq, k=10, ef_search=200)
        recalls_by_efc[efc] = recall_at_k(ids, gt_nbrs, 10)

    # At efC=100, recall should be at least 0.95 (may be lower with some seeds)
    assert recalls_by_efc[100] >= 0.95, (
        f"Dataset B recall at efC=100: {recalls_by_efc[100]:.4f} < 0.95"
    )

    # At efC=400, recall should be at least 0.99
    assert recalls_by_efc[400] >= 0.99, (
        f"Dataset B recall at efC=400: {recalls_by_efc[400]:.4f} < 0.99"
    )

    # Recall should improve or saturate at near-perfect with efC
    assert recalls_by_efc[400] >= recalls_by_efc[100] - 0.005, (
        f"Recall dropped significantly with efC: {recalls_by_efc[100]:.4f} -> {recalls_by_efc[400]:.4f}"
    )


def test_recall_monotonicity_dataset_b():
    """Verify monotonicity on Dataset B."""
    xb, xq, gt_nbrs, _, _ = get_dataset_b()
    n_train = len(xb)
    dim = xb.shape[1]

    idx = HNSWIndex(dim=dim, metric="l2", max_elements=n_train, M=16, ef_construction=100, seed=42)
    idx.add(xb, np.arange(n_train, dtype=np.int64))

    recalls = []
    ef_values = [10, 20, 50, 100, 200]
    for ef in ef_values:
        idx.set_ef_search(ef)
        ids, _ = idx.search(xq, k=10)
        rec = recall_at_k(ids, gt_nbrs, k=10)
        recalls.append(rec)

    for i in range(len(recalls) - 1):
        assert recalls[i+1] >= recalls[i] - 1e-4, (
            f"Recall dropped: ef={ef_values[i]}: {recalls[i]:.4f} -> "
            f"ef={ef_values[i+1]}: {recalls[i+1]:.4f}"
        )


def test_sigma_sensitivity():
    """Verify that recall improves as cluster spread increases."""
    rng_base = np.random.default_rng(42)
    n_train, n_test, dim, n_clusters = 10000, 500, 64, 30

    recalls_by_sigma = {}
    for sigma in [0.1, 0.5, 1.0]:
        rng = np.random.default_rng(42)
        centroids = rng.standard_normal((n_clusters, dim)).astype(np.float32)
        train_labels = rng.integers(0, n_clusters, size=n_train)
        noise_train = rng.standard_normal((n_train, dim)).astype(np.float32) * sigma
        xb = (centroids[train_labels] + noise_train).astype(np.float32)
        test_labels = rng.integers(0, n_clusters, size=n_test)
        noise_test = rng.standard_normal((n_test, dim)).astype(np.float32) * sigma
        xq = (centroids[test_labels] + noise_test).astype(np.float32)
        gt_nbrs, _ = compute_ground_truth_bruteforce(xb, xq, "l2", 10)

        idx = HNSWIndex(dim=dim, metric="l2", max_elements=n_train, M=16, ef_construction=100, seed=42)
        idx.add(xb, np.arange(n_train, dtype=np.int64))
        ids, _ = idx.search(xq, k=10, ef_search=200)
        recalls_by_sigma[sigma] = recall_at_k(ids, gt_nbrs, 10)

    # Wider clusters should give higher recall
    assert recalls_by_sigma[1.0] >= recalls_by_sigma[0.1] - 0.01, (
        f"Wider clusters didn't help: sigma=0.1: {recalls_by_sigma[0.1]:.4f}, "
        f"sigma=1.0: {recalls_by_sigma[1.0]:.4f}"
    )

    # sigma=1.0 should reach near-perfect recall
    assert recalls_by_sigma[1.0] >= 0.99, (
        f"sigma=1.0 recall: {recalls_by_sigma[1.0]:.4f} < 0.99"
    )


def test_self_retrieval_rate_dataset_a():
    """Self-retrieval: searching with xb[i] should return i as rank 0."""
    xb, _, _, _, _ = get_dataset_a()
    n = len(xb)
    idx = HNSWIndex(dim=xb.shape[1], metric="l2", max_elements=n, M=16, ef_construction=100, seed=42)
    idx.add(xb, np.arange(n, dtype=np.int64), num_threads=1)

    def search_fn(q):
        return idx.search(q, k=1, ef_search=200)

    rate = self_retrieval_rate(search_fn, xb[:500])  # Check first 500 for speed
    assert rate >= 0.985, f"Self-retrieval rate: {rate:.4f} < 0.985"
