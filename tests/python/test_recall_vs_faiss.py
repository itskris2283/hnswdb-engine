import numpy as np
import pytest
import faiss
import hnswdb
from hnswdb import HNSWIndex


def generate_clustered_data(n, d, n_clusters=10, seed=42):
    rng = np.random.default_rng(seed)
    centroids = rng.standard_normal((n_clusters, d)).astype(np.float32)
    labels = rng.integers(0, n_clusters, size=n)
    noise = rng.standard_normal((n, d)).astype(np.float32) * 0.1
    data = centroids[labels] + noise
    return data, centroids


def test_recall_vs_faiss_l2():
    n = 2000
    d = 64
    k = 10
    nq = 100

    data, centroids = generate_clustered_data(n, d, n_clusters=10, seed=42)
    # Queries centered around clusters
    q_labels = np.random.default_rng(123).integers(0, 10, size=nq)
    queries = centroids[q_labels] + np.random.default_rng(123).standard_normal((nq, d)).astype(np.float32) * 0.1

    # FAISS ground truth
    faiss_idx = faiss.IndexFlatL2(d)
    faiss_idx.add(data)
    _, faiss_ids = faiss_idx.search(queries, k)

    # HNSWDB Index
    hnsw = HNSWIndex(dim=d, metric="l2", max_elements=n, M=16, ef_construction=100, ef_search=100, seed=42)
    hnsw.add(data, np.arange(n, dtype=np.int64))

    hnsw_ids, _ = hnsw.search(queries, k=k, ef_search=100)

    # Compute recall@k
    recalls = []
    for i in range(nq):
        gt_set = set(faiss_ids[i])
        found_set = set(hnsw_ids[i])
        recalls.append(len(gt_set & found_set) / k)
    mean_recall = float(np.mean(recalls))
    assert mean_recall >= 0.95, f"Recall@10 {mean_recall} is below 0.95"


def test_recall_vs_faiss_ip():
    n = 2000
    d = 64
    k = 10
    nq = 100

    data, centroids = generate_clustered_data(n, d, n_clusters=10, seed=42)
    # Normalize for stable IP
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    data = (data / np.maximum(norms, 1e-12)).astype(np.float32)

    q_labels = np.random.default_rng(123).integers(0, 10, size=nq)
    queries = centroids[q_labels] + np.random.default_rng(123).standard_normal((nq, d)).astype(np.float32) * 0.1
    q_norms = np.linalg.norm(queries, axis=1, keepdims=True)
    queries = (queries / np.maximum(q_norms, 1e-12)).astype(np.float32)

    # FAISS ground truth
    faiss_idx = faiss.IndexFlatIP(d)
    faiss_idx.add(data)
    _, faiss_ids = faiss_idx.search(queries, k)

    # HNSWDB Index
    hnsw = HNSWIndex(dim=d, metric="ip", max_elements=n, M=16, ef_construction=100, ef_search=100, seed=42)
    hnsw.add(data, np.arange(n, dtype=np.int64))

    hnsw_ids, _ = hnsw.search(queries, k=k, ef_search=100)

    recalls = []
    for i in range(nq):
        gt_set = set(faiss_ids[i])
        found_set = set(hnsw_ids[i])
        recalls.append(len(gt_set & found_set) / k)
    mean_recall = float(np.mean(recalls))
    assert mean_recall >= 0.95, f"Recall@10 {mean_recall} is below 0.95"


def test_recall_vs_faiss_cosine():
    n = 2000
    d = 64
    k = 10
    nq = 100

    data, centroids = generate_clustered_data(n, d, n_clusters=10, seed=42)
    q_labels = np.random.default_rng(123).integers(0, 10, size=nq)
    queries = (centroids[q_labels] + np.random.default_rng(123).standard_normal((nq, d)).astype(np.float32) * 0.1).astype(np.float32)

    # FAISS Cosine = normalize + IndexFlatIP
    data_norm = data / np.maximum(np.linalg.norm(data, axis=1, keepdims=True), 1e-12)
    q_norm = queries / np.maximum(np.linalg.norm(queries, axis=1, keepdims=True), 1e-12)

    faiss_idx = faiss.IndexFlatIP(d)
    faiss_idx.add(data_norm.astype(np.float32))
    _, faiss_ids = faiss_idx.search(q_norm.astype(np.float32), k)

    hnsw = HNSWIndex(dim=d, metric="cosine", max_elements=n, M=16, ef_construction=100, ef_search=100, seed=42)
    hnsw.add(data, np.arange(n, dtype=np.int64))

    hnsw_ids, _ = hnsw.search(queries, k=k, ef_search=100)

    recalls = []
    for i in range(nq):
        gt_set = set(faiss_ids[i])
        found_set = set(hnsw_ids[i])
        recalls.append(len(gt_set & found_set) / k)
    mean_recall = float(np.mean(recalls))
    assert mean_recall >= 0.95, f"Recall@10 {mean_recall} is below 0.95"


def test_ef_search_monotonicity():
    n = 2000
    d = 32
    k = 10
    nq = 50

    data, centroids = generate_clustered_data(n, d, n_clusters=10, seed=42)
    q_labels = np.random.default_rng(123).integers(0, 10, size=nq)
    queries = centroids[q_labels] + np.random.default_rng(123).standard_normal((nq, d)).astype(np.float32) * 0.1

    faiss_idx = faiss.IndexFlatL2(d)
    faiss_idx.add(data)
    _, faiss_ids = faiss_idx.search(queries, k)

    hnsw = HNSWIndex(dim=d, metric="l2", max_elements=n, M=16, ef_construction=100, ef_search=10, seed=42)
    hnsw.add(data, np.arange(n, dtype=np.int64))

    ef_values = [10, 20, 50, 100, 200]
    recalls = []
    for ef in ef_values:
        hnsw_ids, _ = hnsw.search(queries, k=k, ef_search=ef)
        cur_recalls = []
        for i in range(nq):
            gt_set = set(faiss_ids[i])
            found_set = set(hnsw_ids[i])
            cur_recalls.append(len(gt_set & found_set) / k)
        recalls.append(float(np.mean(cur_recalls)))

    # Verify overall monotonicity trend
    for i in range(len(recalls) - 1):
        assert recalls[i] <= recalls[i + 1] + 0.01, f"Recall at ef={ef_values[i]} ({recalls[i]}) was higher than at ef={ef_values[i+1]} ({recalls[i+1]})"
    assert recalls[-1] >= 0.95


def test_clustered_dataset_recall_floor():
    """Guards recall floor on clustered synthetic dataset (N=5000, d=32, 20 clusters)
    under default construction options (M=16, ef_c=100, ef_search=100).
    """
    n = 5000
    d = 32
    k = 10
    nq = 200

    data, centroids = generate_clustered_data(n, d, n_clusters=20, seed=42)
    q_labels = np.random.default_rng(999).integers(0, 20, size=nq)
    queries = centroids[q_labels] + np.random.default_rng(999).standard_normal((nq, d)).astype(np.float32) * 0.05

    faiss_idx = faiss.IndexFlatL2(d)
    faiss_idx.add(data)
    _, faiss_ids = faiss_idx.search(queries, k)

    hnsw = HNSWIndex(dim=d, metric="l2", max_elements=n, M=16, ef_construction=100, ef_search=100, seed=42)
    hnsw.add(data, np.arange(n, dtype=np.int64))

    hnsw_ids, _ = hnsw.search(queries, k=k, ef_search=100)

    recalls = []
    for i in range(nq):
        gt_set = set(faiss_ids[i])
        found_set = set(hnsw_ids[i])
        recalls.append(len(gt_set & found_set) / k)
    mean_recall = float(np.mean(recalls))
    assert mean_recall >= 0.990, f"Clustered dataset recall floor: achieved {mean_recall:.4f}, expected >= 0.990"


