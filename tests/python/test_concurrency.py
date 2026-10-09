import concurrent.futures
import time
import numpy as np
import pytest
from hnswdb import HNSWIndex


def test_concurrent_python_readers_and_writers():
    dim = 32
    max_elements = 2000
    idx = HNSWIndex(dim=dim, metric="l2", max_elements=max_elements, M=16, ef_construction=100, ef_search=50, seed=42)

    # Prepopulate with 100 items
    init_v = np.random.randn(100, dim).astype(np.float32)
    init_ids = np.arange(100, dtype=np.int64)
    idx.add(init_v, init_ids)

    stop_flag = False

    def writer_task(writer_id, count=100):
        rng = np.random.default_rng(1000 + writer_id)
        for i in range(count):
            label = 1000 + writer_id * 1000 + i
            v = rng.standard_normal(dim).astype(np.float32)
            idx.add(v, np.array([label], dtype=np.int64))
            time.sleep(0.001)

    def reader_task(reader_id, queries_count=200):
        rng = np.random.default_rng(5000 + reader_id)
        for _ in range(queries_count):
            if stop_flag:
                break
            q = rng.standard_normal((1, dim)).astype(np.float32)
            ids, dists = idx.search(q, k=5)
            assert ids.shape == (1, 5)
            assert dists.shape == (1, 5)
            time.sleep(0.001)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        # Launch 4 writers and 4 readers
        writer_futures = [executor.submit(writer_task, wid, 50) for wid in range(4)]
        reader_futures = [executor.submit(reader_task, rid, 100) for rid in range(4)]

        # Wait for all writers to finish
        for f in concurrent.futures.as_completed(writer_futures):
            f.result()

        stop_flag = True

        for f in concurrent.futures.as_completed(reader_futures):
            f.result()

    assert len(idx) == 100 + 4 * 50


def test_concurrent_batch_threading():
    dim = 32
    n = 500
    idx = HNSWIndex(dim=dim, metric="l2", max_elements=1000)
    vecs = np.random.randn(n, dim).astype(np.float32)
    ids = np.arange(n, dtype=np.int64)

    # Multi-threaded add_batch
    idx.add(vecs, ids, num_threads=4)
    assert len(idx) == n

    # Multi-threaded search_batch
    queries = np.random.randn(50, dim).astype(np.float32)
    res_ids, res_dists = idx.search(queries, k=10, num_threads=4)
    assert res_ids.shape == (50, 10)
