"""
Regression tests for graph invariants and construction quality.
Tests verify that the graph has no self-loops or duplicate links,
maintains minimum connectivity, and achieves expected recall.
"""

import numpy as np
import pytest
from hnswdb import HNSWIndex
from benchmarks.datasets_synth import get_dataset_a
from benchmarks.metrics import recall_at_k


def test_graph_invariants_dataset_a():
    """Verify graph topology invariants: no self-loops, no duplicate links,
    all nodes have degree >= 1 and <= M0."""
    xb, xq, gt_nbrs, _, _ = get_dataset_a()
    n_train = len(xb)
    dim = xb.shape[1]

    idx = HNSWIndex(dim=dim, metric="l2", max_elements=n_train, M=16, ef_construction=100, seed=42)
    idx.add(xb, np.arange(n_train, dtype=np.int64))

    diag = idx.diagnostics()
    assert diag["duplicate_links_count"] == 0.0, (
        f"Found {diag['duplicate_links_count']:.0f} duplicate links in graph"
    )
    assert diag["self_links_count"] == 0.0, (
        f"Found {diag['self_links_count']:.0f} self-links in graph"
    )
    assert diag["layer0_degree_max"] <= 32.0, (
        f"Max degree {diag['layer0_degree_max']:.0f} exceeds M0=32"
    )
    assert diag["layer0_degree_min"] >= 1.0, (
        f"Min degree {diag['layer0_degree_min']:.0f} < 1 (isolated nodes)"
    )


def test_recall_dataset_a():
    """Verify recall@10 on Dataset A (easy distribution, 5K x 32d)."""
    xb, xq, gt_nbrs, _, _ = get_dataset_a()
    n_train = len(xb)
    dim = xb.shape[1]

    idx = HNSWIndex(dim=dim, metric="l2", max_elements=n_train, M=16, ef_construction=100, seed=42)
    idx.add(xb, np.arange(n_train, dtype=np.int64))

    ids, _ = idx.search(xq, k=10, ef_search=50)
    rec = recall_at_k(ids, gt_nbrs, k=10)
    assert rec >= 0.990, f"Recall@10 {rec:.4f} is below 0.990 on Dataset A at ef=50"


def test_graph_connectivity():
    """Verify that the graph is weakly connected (single component)."""
    xb, _, _, _, _ = get_dataset_a()
    n_train = len(xb)

    idx = HNSWIndex(dim=xb.shape[1], metric="l2", max_elements=n_train, M=16, ef_construction=100, seed=42)
    idx.add(xb, np.arange(n_train, dtype=np.int64))

    diag = idx.diagnostics()
    assert diag["weakly_connected_components"] == 1.0, (
        f"Graph has {diag['weakly_connected_components']:.0f} components (expected 1)"
    )


def test_approximation_ratio_correct():
    """Verify approximation ratio is close to 1.0 (not inflated by squared distances)."""
    from benchmarks.metrics import approx_ratio

    xb, xq, gt_nbrs, gt_dists, _ = get_dataset_a()
    n_train = len(xb)

    idx = HNSWIndex(dim=xb.shape[1], metric="l2", max_elements=n_train, M=16, ef_construction=100, seed=42)
    idx.add(xb, np.arange(n_train, dtype=np.int64))

    ids, dists = idx.search(xq, k=10, ef_search=50)
    ar = approx_ratio(dists, gt_dists, k=10, squared=True)

    # AR should be between 1.0 and 1.05 (close to exact)
    assert 1.0 <= ar <= 1.10, f"Approximation ratio {ar:.4f} out of expected range [1.0, 1.10]"
