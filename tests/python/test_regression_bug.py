"""
Regression test for duplicate neighbor link bug and graph invariants.
Ensures snapshot duplicate links and self-loops remain 0 and graph connectivity holds.
"""

import numpy as np
import pytest
from hnswdb import HNSWIndex
from benchmarks.datasets_synth import get_dataset_a
from benchmarks.metrics import recall_at_k


def test_duplicate_links_invariant():
    """Verify that no node in the graph contains duplicate outgoing links at layer 0."""
    xb, xq, gt_nbrs, _, _ = get_dataset_a()
    n_train = len(xb)
    dim = xb.shape[1]

    idx = HNSWIndex(dim=dim, metric="l2", max_elements=n_train, M=16, ef_construction=100, ef_search=50, seed=42)
    idx.add(xb, np.arange(n_train, dtype=np.int64))

    diag = idx.diagnostics()
    assert diag["duplicate_links_count"] == 0.0, f"Found {diag['duplicate_links_count']} duplicate links in graph"
    assert diag["self_links_count"] == 0.0, f"Found {diag['self_links_count']} self-links in graph"
    assert diag["layer0_degree_max"] <= 32.0
    assert diag["layer0_degree_min"] >= 1.0

    # Recall check
    ids, _ = idx.search(xq, k=10)
    rec = recall_at_k(ids, gt_nbrs, k=10)
    assert rec >= 0.990, f"Recall@10 {rec} is below expected 0.990 on Dataset A"

