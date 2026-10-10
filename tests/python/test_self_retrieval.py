"""
Corrected self-retrieval unit tests.
Verifies:
1. Exact index gives 1.0 (100% self-retrieval rate).
2. Index missing known points gives the expected fraction.
3. Duplicate rows (identical coordinates) handled by distance-0 equality.
"""

import numpy as np
import pytest
import hnswdb
from benchmarks.metrics import self_retrieval_rate


def test_exact_index_self_retrieval_rate_one():
    """An index queried with its own inserted vectors must achieve 1.0 self-retrieval."""
    dim = 32
    n = 200
    rng = np.random.default_rng(42)
    xb = rng.standard_normal((n, dim)).astype(np.float32)
    ids = np.arange(n, dtype=np.int64)

    idx = hnswdb.HNSWIndex(dim=dim, metric="l2", max_elements=n, M=16, ef_construction=100, seed=42)
    idx.add(xb, ids, num_threads=1)

    def search_fn(q):
        return idx.search(q, k=1, ef_search=50, num_threads=1)

    rate = self_retrieval_rate(search_fn, xb, ids)
    assert rate == 1.0, f"Expected 1.0 self-retrieval on exact index, got {rate}"


def test_missing_points_self_retrieval_expected_fraction():
    """Querying with a set where 20% of points are not in the index yields exactly 0.80."""
    dim = 32
    n_indexed = 80
    n_missing = 20
    n_total = n_indexed + n_missing
    rng = np.random.default_rng(123)

    xb_total = rng.standard_normal((n_total, dim)).astype(np.float32)
    ids_total = np.arange(n_total, dtype=np.int64)

    # Index only the first 80 points
    idx = hnswdb.HNSWIndex(dim=dim, metric="l2", max_elements=n_indexed, M=16, ef_construction=100, seed=42)
    idx.add(xb_total[:n_indexed], ids_total[:n_indexed], num_threads=1)

    def search_fn(q):
        return idx.search(q, k=1, ef_search=50, num_threads=1)

    # Query with all 100 points
    rate = self_retrieval_rate(search_fn, xb_total, ids_total)
    assert rate == pytest.approx(0.80, abs=1e-5), f"Expected 0.80 self-retrieval rate, got {rate}"


def test_duplicate_rows_handled_by_distance_zero():
    """Duplicate rows (identical coordinates with different labels) must be handled by distance-0 equality."""
    dim = 16
    rng = np.random.default_rng(77)
    base_vectors = rng.standard_normal((50, dim)).astype(np.float32)

    # Create duplicates: vector 0 is cloned at position 50 with label 50
    # vector 1 is cloned at position 51 with label 51
    xb = np.vstack([base_vectors, base_vectors[:2]]).astype(np.float32)
    ids = np.arange(len(xb), dtype=np.int64)

    idx = hnswdb.HNSWIndex(dim=dim, metric="l2", max_elements=len(xb), M=16, ef_construction=100, seed=42)
    idx.add(xb, ids, num_threads=1)

    # When querying with vector 0, top result can be label 0 or label 50 (both distance == 0.0)
    q0 = np.ascontiguousarray(xb[0:1], dtype=np.float32)
    found_ids, dists = idx.search(q0, k=2, ef_search=50, num_threads=1)

    # Distance to top neighbor must be exactly 0.0
    assert dists[0][0] == pytest.approx(0.0, abs=1e-6)
    # The returned label must be either 0 or 50
    assert found_ids[0][0] in (0, 50)
