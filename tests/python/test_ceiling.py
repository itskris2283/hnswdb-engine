"""
Tests for recall convergence, monotonicity, and cluster bridge characteristics.
Verifies behavior on clustered synthetic distributions and real datasets.
"""

import numpy as np
import pytest
from hnswdb import HNSWIndex
from benchmarks.datasets_synth import get_dataset_b, get_dataset_c
from benchmarks.metrics import recall_at_k, self_retrieval_rate


def test_recall_monotonicity_and_convergence_dataset_b():
    """Verify that recall is non-decreasing with ef_search on Dataset B."""
    xb, xq, gt_nbrs, _, _ = get_dataset_b()
    n_train = len(xb)
    dim = xb.shape[1]

    idx = HNSWIndex(dim=dim, metric="l2", max_elements=n_train, M=16, ef_construction=100, ef_search=50, seed=42)
    idx.add(xb, np.arange(n_train, dtype=np.int64))

    recalls = []
    for ef in [10, 20, 50, 100, 200]:
        idx.set_ef_search(ef)
        ids, _ = idx.search(xq, k=10)
        rec = recall_at_k(ids, gt_nbrs, k=10)
        recalls.append(rec)

    # Monotonicity check
    for i in range(len(recalls) - 1):
        assert recalls[i+1] >= recalls[i] - 1e-4, f"Recall dropped from ef={recalls[i]} to {recalls[i+1]}"

    # Convergence floor on Dataset B (20k x 64d)
    assert recalls[-1] >= 0.970, f"Final recall {recalls[-1]} is below expected 0.970"


def test_fashion_mnist_convergence():
    """Verify that Fashion-MNIST reaches high recall at ef=100 with default parameters."""
    xb, xq, _, _, _ = get_dataset_c()
    n_train = 5000
    n_test = 200
    dim = xb.shape[1]

    from benchmarks.datasets_synth import compute_ground_truth_bruteforce
    gt_nbrs, _ = compute_ground_truth_bruteforce(xb[:n_train], xq[:n_test], metric="l2", k=10)

    idx = HNSWIndex(dim=dim, metric="l2", max_elements=n_train, M=16, ef_construction=100, ef_search=50, seed=42)
    idx.add(xb[:n_train], np.arange(n_train, dtype=np.int64))

    idx.set_ef_search(100)
    ids, _ = idx.search(xq[:n_test], k=10)
    rec = recall_at_k(ids, gt_nbrs, k=10)
    assert rec >= 0.980, f"Recall {rec} on Fashion-MNIST subset below 0.980"

