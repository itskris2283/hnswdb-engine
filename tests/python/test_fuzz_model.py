import numpy as np
import pytest
from hnswdb import HNSWIndex, FlatIndex


def test_fuzz_operations():
    dim = 16
    capacity = 500
    rng = np.random.default_rng(42)

    hnsw = HNSWIndex(dim=dim, metric="l2", max_elements=capacity, M=16, ef_construction=100, ef_search=50, seed=42)
    # Python dict ground truth: label -> vector
    model = {}

    ops = ["add", "upsert", "delete", "search", "contains", "get_vector", "compact"]
    op_weights = [0.4, 0.15, 0.15, 0.2, 0.05, 0.04, 0.01]

    num_ops = 3000
    next_label = 1

    for step in range(num_ops):
        op = rng.choice(ops, p=op_weights)

        # If getting close to capacity, compact or resize
        if hnsw.stats()["total_elements"] >= capacity - 30:
            if hnsw.stats()["tombstone_ratio"] > 0.2:
                hnsw = hnsw.compact()
            else:
                capacity += 500
                hnsw.resize(capacity)

        if op == "add":
            if len(model) < capacity - 50:
                batch_size = int(rng.integers(1, 10))
                vecs = rng.standard_normal((batch_size, dim)).astype(np.float32)
                labels = np.arange(next_label, next_label + batch_size, dtype=np.int64)
                next_label += batch_size

                hnsw.add(vecs, labels)
                for l, v in zip(labels, vecs):
                    model[int(l)] = v

        elif op == "upsert":
            if model and hnsw.stats()["total_elements"] < capacity - 5:
                # pick existing label
                l = int(rng.choice(list(model.keys())))
                v = rng.standard_normal(dim).astype(np.float32)
                hnsw.add(v, np.array([l], dtype=np.int64))
                model[l] = v


        elif op == "delete":
            if model and len(model) > 5:
                del_count = int(rng.integers(1, min(5, len(model))))
                chosen = [int(x) for x in rng.choice(list(model.keys()), size=del_count, replace=False)]
                hnsw.delete(chosen)
                for l in chosen:
                    del model[l]

        elif op == "contains":
            if model:
                l_present = int(rng.choice(list(model.keys())))
                assert hnsw.contains(l_present)
                l_absent = next_label + 1000
                assert not hnsw.contains(l_absent)

        elif op == "get_vector":
            if model:
                l = int(rng.choice(list(model.keys())))
                v_expected = model[l]
                v_actual = hnsw.get_vector(l)
                assert np.allclose(v_expected, v_actual, atol=1e-5)

        elif op == "search":
            if len(model) >= 5:
                q = rng.standard_normal((1, dim)).astype(np.float32)
                k = min(5, len(model))
                ids, dists = hnsw.search(q, k=k)
                assert ids.shape == (1, k)
                # Results must be in live model
                for found_id in ids[0]:
                    if found_id != -1:
                        assert found_id in model

        elif op == "compact":
            if len(model) > 10:
                hnsw = hnsw.compact()
                assert len(hnsw) == len(model)

        # Invariant check
        assert len(hnsw) == len(model)

    # Final full check
    assert len(hnsw) == len(model)
    for l, v in model.items():
        assert hnsw.contains(l)
        assert np.allclose(v, hnsw.get_vector(l), atol=1e-5)


def test_churn_delete_20pct_insert_20pct_recall():
    """Churn test: delete 20%, insert 20% new vectors, verify recall vs recomputed exact truth."""
    dim = 32
    n = 1000
    k = 10
    nq = 100
    rng = np.random.default_rng(42)

    xb = rng.standard_normal((n, dim)).astype(np.float32)
    ids = np.arange(n, dtype=np.int64)
    xq = rng.standard_normal((nq, dim)).astype(np.float32)

    idx = HNSWIndex(dim=dim, metric="l2", max_elements=1500, M=16, ef_construction=100, seed=42)
    idx.add(xb, ids, num_threads=1)

    # 1. Delete 20% (200 random IDs)
    del_count = 200
    del_ids = rng.choice(ids, size=del_count, replace=False).tolist()
    idx.delete(del_ids)
    del_set = set(del_ids)

    # 2. Insert 20% (200 new vectors with new IDs)
    new_xb = rng.standard_normal((del_count, dim)).astype(np.float32)
    new_ids = np.arange(n, n + del_count, dtype=np.int64)
    idx.add(new_xb, new_ids, num_threads=1)

    # 3. Construct current active dataset and recompute exact brute force truth
    active_mask = ~np.isin(ids, del_ids)
    active_xb = np.vstack([xb[active_mask], new_xb])
    active_ids = np.concatenate([ids[active_mask], new_ids])

    gt_nbrs = np.zeros((nq, k), dtype=np.int64)
    for q in range(nq):
        diff = active_xb - xq[q]
        dists = np.sum(diff * diff, axis=1)
        top_k = np.argsort(dists)[:k]
        gt_nbrs[q] = active_ids[top_k]

    # 4. Search HNSWDB at ef_search=100
    found_ids, _ = idx.search(xq, k=k, ef_search=100, num_threads=1)

    # 5. Assertions:
    # A) No deleted IDs returned
    for row in found_ids:
        for found_id in row:
            assert found_id not in del_set, f"Deleted ID {found_id} returned in search results!"

    # B) Recall vs recomputed exact truth >= 0.98
    recalls = [len(set(f) & set(t)) / k for f, t in zip(found_ids, gt_nbrs)]
    mean_recall = float(np.mean(recalls))
    assert mean_recall >= 0.98, f"Post-churn recall {mean_recall:.4f} < 0.98"

