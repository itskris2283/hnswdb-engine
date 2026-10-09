import numpy as np
import pytest
import hnswdb
from hnswdb import (
    HNSWIndex,
    FlatIndex,
    HnswdbError,
    DimensionMismatch,
    InvalidVector,
    CapacityExceeded,
    InvalidArgument,
)


def test_basic_lifecycle():
    dim = 64
    idx = HNSWIndex(dim=dim, metric="l2", max_elements=100, M=16, ef_construction=100, ef_search=50, seed=42)
    assert len(idx) == 0
    assert idx.size() == 0
    assert idx.capacity() == 100
    assert idx.dim() == dim

    # Add single vector
    v = np.random.randn(dim).astype(np.float32)
    idx.add(v, np.array([10], dtype=np.int64))
    assert len(idx) == 1
    assert idx.contains(10)
    assert not idx.contains(999)

    fetched = idx.get_vector(10)
    assert np.allclose(v, fetched, atol=1e-6)

    # Search
    ids, dists = idx.search(v, k=5)
    assert ids.shape == (1, 5)
    assert dists.shape == (1, 5)
    assert ids[0, 0] == 10
    assert dists[0, 0] == pytest.approx(0.0, abs=1e-5)


def test_batch_add_and_search_shapes():
    dim = 32
    idx = HNSWIndex(dim=dim, metric="l2", max_elements=500)
    
    n = 100
    vectors = np.random.randn(n, dim).astype(np.float32)
    ids = np.arange(n, dtype=np.int64)

    idx.add(vectors, ids)
    assert len(idx) == n

    nq = 10
    queries = np.random.randn(nq, dim).astype(np.float32)
    k = 7
    res_ids, res_dists = idx.search(queries, k=k)

    assert res_ids.shape == (nq, k)
    assert res_dists.shape == (nq, k)
    assert res_ids.dtype == np.int64
    assert res_dists.dtype == np.float32


def test_1d_inputs():
    dim = 16
    idx = HNSWIndex(dim=dim, metric="l2", max_elements=50)

    # 1D vector and 1D id
    v1 = np.ones(dim, dtype=np.float32)
    idx.add(v1, np.array([1], dtype=np.int64))

    # 1D search
    q = np.ones(dim, dtype=np.float32)
    ids, dists = idx.search(q, k=3)
    assert ids.shape == (1, 3)
    assert ids[0, 0] == 1
    assert dists[0, 0] == pytest.approx(0.0, abs=1e-5)
    assert ids[0, 1] == -1
    assert np.isinf(dists[0, 1])


def test_type_casting():
    dim = 8
    idx = HNSWIndex(dim=dim, metric="l2", max_elements=50)

    # float64 input should be automatically cast to float32
    v = np.random.randn(5, dim).astype(np.float64)
    ids = np.array([1, 2, 3, 4, 5], dtype=np.int32)  # int32 should cast to int64

    idx.add(v, ids)
    assert len(idx) == 5

    q = np.random.randn(2, dim).astype(np.float64)
    res_ids, res_dists = idx.search(q, k=2)
    assert res_ids.shape == (2, 2)


def test_dimension_mismatch():
    dim = 16
    idx = HNSWIndex(dim=dim, metric="l2", max_elements=50)

    bad_vec = np.random.randn(10, dim + 1).astype(np.float32)
    with pytest.raises(DimensionMismatch):
        idx.add(bad_vec, np.arange(10, dtype=np.int64))

    bad_query = np.random.randn(2, dim - 1).astype(np.float32)
    with pytest.raises(DimensionMismatch):
        idx.search(bad_query, k=5)


def test_invalid_vector_exceptions():
    dim = 16
    idx = HNSWIndex(dim=dim, metric="l2", max_elements=50)

    # NaN vector
    nan_vec = np.zeros(dim, dtype=np.float32)
    nan_vec[0] = np.nan
    with pytest.raises(InvalidVector):
        idx.add(nan_vec, np.array([1], dtype=np.int64))

    # Inf vector
    inf_vec = np.zeros(dim, dtype=np.float32)
    inf_vec[3] = np.inf
    with pytest.raises(InvalidVector):
        idx.add(inf_vec, np.array([2], dtype=np.int64))

    # Cosine zero vector
    cos_idx = HNSWIndex(dim=dim, metric="cosine", max_elements=50)
    zero_vec = np.zeros(dim, dtype=np.float32)
    with pytest.raises(InvalidVector):
        cos_idx.add(zero_vec, np.array([3], dtype=np.int64))


def test_capacity_exceeded():
    dim = 8
    idx = HNSWIndex(dim=dim, metric="l2", max_elements=10)

    vecs = np.random.randn(10, dim).astype(np.float32)
    idx.add(vecs, np.arange(10, dtype=np.int64))

    extra_vec = np.random.randn(1, dim).astype(np.float32)
    with pytest.raises(CapacityExceeded):
        idx.add(extra_vec, np.array([11], dtype=np.int64))

    # Resize allows adding more
    idx.resize(20)
    assert idx.capacity() == 20
    idx.add(extra_vec, np.array([11], dtype=np.int64))
    assert len(idx) == 11


def test_stats_dictionary():
    dim = 16
    idx = HNSWIndex(dim=dim, metric="ip", max_elements=100)
    vecs = np.random.randn(20, dim).astype(np.float32)
    idx.add(vecs, np.arange(20, dtype=np.int64))

    st = idx.stats()
    assert isinstance(st, dict)
    assert st["size"] == 20
    assert st["capacity"] == 100
    assert st["tombstones"] == 0
    assert "max_level" in st
    assert "tombstone_ratio" in st
    assert st["metric"] == "ip"
    assert "kernel" in st
    assert "entry_point" in st
    assert st["needs_compaction"] is False
    assert st["memory_bytes"] > 0


def test_flat_index_api():
    dim = 16
    flat = FlatIndex(dim=dim, metric="l2")
    assert len(flat) == 0

    v = np.random.randn(10, dim).astype(np.float32)
    ids = np.arange(10, dtype=np.int64)
    flat.add(v, ids)
    assert len(flat) == 10
    assert flat.contains(5)

    q = v[3:5]
    res_ids, res_dists = flat.search(q, k=3)
    assert res_ids[0, 0] == 3
    assert res_ids[1, 0] == 4
    assert res_dists[0, 0] == pytest.approx(0.0, abs=1e-5)

    # Delete
    flat.delete(3)
    assert not flat.contains(3)
    assert len(flat) == 9


def test_diagnostics_dictionary():
    dim = 16
    idx = HNSWIndex(dim=dim, metric="l2", max_elements=50, M=16, ef_construction=100, ef_search=50, seed=42)
    vecs = np.random.randn(30, dim).astype(np.float32)
    idx.add(vecs, np.arange(30, dtype=np.int64))

    diag = idx.diagnostics()
    assert isinstance(diag, dict)
    assert diag["unreachable_nodes_count"] == 0
    assert diag["zero_in_degree_count"] == 0
    assert diag["duplicate_links_count"] == 0
    assert diag["self_links_count"] == 0
    assert diag["weakly_connected_components"] == 1
    assert diag["layer0_degree_min"] >= 1
    assert diag["layer0_degree_mean"] > 1.0
    assert diag["layer0_degree_max"] <= 32
    assert diag["reciprocal_links_fraction"] > 0.0
    assert diag["distance_computations_build"] > 0



