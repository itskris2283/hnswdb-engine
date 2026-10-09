import numpy as np
import pytest
from hnswdb import HNSWIndex


def test_simple_equality_filter():
    dim = 16
    n = 100
    rng = np.random.default_rng(42)

    vecs = rng.standard_normal((n, dim)).astype(np.float32)
    ids = np.arange(n, dtype=np.int64)
    metadata = [{"category": "shoes" if i % 2 == 0 else "clothing", "price": int(i)} for i in range(n)]

    idx = HNSWIndex(dim=dim, metric="l2", max_elements=200, seed=42)
    idx.add(vecs, ids, metadata=metadata)

    # Search only shoes
    q = vecs[0:1]
    res_ids, res_dists = idx.search(q, k=10, filter={"category": "shoes"})
    for fid in res_ids[0]:
        if fid != -1:
            meta = idx.get_metadata(fid)
            assert meta["category"] == "shoes"


def test_complex_filter_operators():
    dim = 16
    n = 200
    rng = np.random.default_rng(42)

    vecs = rng.standard_normal((n, dim)).astype(np.float32)
    ids = np.arange(n, dtype=np.int64)
    metadata = [
        {
            "brand": ["apple", "google", "meta", "amazon"][i % 4],
            "price": int(i * 10),
            "rating": float(i % 5) + 1.0,
            "in_stock": (i % 3 != 0),
        }
        for i in range(n)
    ]

    idx = HNSWIndex(dim=dim, metric="l2", max_elements=500, seed=42)
    idx.add(vecs, ids, metadata=metadata)

    # 1. $and with $in and $lt
    filt1 = {
        "$and": [
            {"brand": {"$in": ["apple", "google"]}},
            {"price": {"$lt": 500}},
        ]
    }
    q = rng.standard_normal((1, dim)).astype(np.float32)
    res_ids, _ = idx.search(q, k=10, filter=filt1)
    for fid in res_ids[0]:
        if fid != -1:
            m = idx.get_metadata(fid)
            assert m["brand"] in ["apple", "google"]
            assert m["price"] < 500

    # 2. $or with $gte
    filt2 = {
        "$or": [
            {"brand": "meta"},
            {"price": {"$gte": 1800}},
        ]
    }
    res_ids, _ = idx.search(q, k=10, filter=filt2)
    for fid in res_ids[0]:
        if fid != -1:
            m = idx.get_metadata(fid)
            assert m["brand"] == "meta" or m["price"] >= 1800

    # 3. $not
    filt3 = {
        "$not": {"brand": "amazon"}
    }
    res_ids, _ = idx.search(q, k=10, filter=filt3)
    for fid in res_ids[0]:
        if fid != -1:
            m = idx.get_metadata(fid)
            assert m["brand"] != "amazon"


def test_filtered_search_recall():
    dim = 32
    n = 1000
    rng = np.random.default_rng(42)

    vecs = rng.standard_normal((n, dim)).astype(np.float32)
    ids = np.arange(n, dtype=np.int64)
    # 5% selective category
    metadata = [{"group": "target" if i < 50 else "other", "val": int(i)} for i in range(n)]

    idx = HNSWIndex(dim=dim, metric="l2", max_elements=1500, M=16, ef_construction=100, ef_search=100, seed=42)
    idx.add(vecs, ids, metadata=metadata)

    q = vecs[0:1]
    res_ids, res_dists = idx.search(q, k=5, filter={"group": "target"})
    assert res_ids[0, 0] == 0
    assert res_dists[0, 0] == pytest.approx(0.0, abs=1e-5)
    for fid in res_ids[0]:
        assert fid < 50
