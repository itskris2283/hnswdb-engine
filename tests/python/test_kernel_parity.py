import numpy as np
import pytest
import hnswdb
from hnswdb import HNSWIndex, FlatIndex, active_kernel_name, set_force_scalar


def test_kernel_parity_l2_and_ip():
    dim = 128
    n = 100
    rng = np.random.default_rng(42)

    vecs = rng.standard_normal((n, dim)).astype(np.float32)
    ids = np.arange(n, dtype=np.int64)
    queries = rng.standard_normal((10, dim)).astype(np.float32)

    # 1. Run with AVX2
    set_force_scalar(False)
    idx_avx = FlatIndex(dim=dim, metric="l2")
    idx_avx.add(vecs, ids)
    avx_ids, avx_dists = idx_avx.search(queries, k=10)

    # 2. Force Scalar
    set_force_scalar(True)
    idx_scalar = FlatIndex(dim=dim, metric="l2")
    idx_scalar.add(vecs, ids)
    scalar_ids, scalar_dists = idx_scalar.search(queries, k=10)

    # Reset force_scalar
    set_force_scalar(False)

    # Compare
    assert np.array_equal(avx_ids, scalar_ids)
    assert np.allclose(avx_dists, scalar_dists, atol=1e-5)



def test_odd_dimensions_parity():
    # Odd dims to test remainder/tail loop
    for d in [1, 7, 15, 33, 99]:
        n = 50
        rng = np.random.default_rng(d)
        vecs = rng.standard_normal((n, d)).astype(np.float32)
        ids = np.arange(n, dtype=np.int64)
        queries = rng.standard_normal((5, d)).astype(np.float32)

        set_force_scalar(False)
        idx1 = FlatIndex(dim=d, metric="ip")
        idx1.add(vecs, ids)
        ids1, dists1 = idx1.search(queries, k=5)

        set_force_scalar(True)
        idx2 = FlatIndex(dim=d, metric="ip")
        idx2.add(vecs, ids)
        ids2, dists2 = idx2.search(queries, k=5)

        set_force_scalar(False)


        assert np.array_equal(ids1, ids2)
        assert np.allclose(dists1, dists2, atol=1e-5)
