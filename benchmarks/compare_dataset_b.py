import time
import numpy as np
import faiss
import hnswlib
import hnswdb
from benchmarks.datasets_synth import get_dataset_b
from benchmarks.metrics import recall_at_k

xb, xq, gt_nbrs, gt_dists, metric = get_dataset_b()
n_train, dim = xb.shape
print(f"Dataset B: {n_train} train, {len(xq)} test, {dim} dim")

# 1. HNSWDB
idx = hnswdb.HNSWIndex(dim=dim, metric="l2", max_elements=n_train, M=16, ef_construction=100, ef_search=50, seed=42)
t0 = time.perf_counter()
idx.add(xb, np.arange(n_train, dtype=np.int64))
t_build_hnswdb = time.perf_counter() - t0
print(f"HNSWDB build: {t_build_hnswdb:.3f}s")
print("HNSWDB diag:", idx.diagnostics())

for ef in [10, 20, 50, 100, 200]:
    idx.set_ef_search(ef)
    ids, _ = idx.search(xq, k=10)
    rec = recall_at_k(ids, gt_nbrs, k=10)
    print(f"  HNSWDB ef={ef}: recall@10 = {rec:.4f}")

# 2. hnswlib
h_idx = hnswlib.Index(space="l2", dim=dim)
h_idx.init_index(max_elements=n_train, ef_construction=100, M=16, random_seed=42)
t0 = time.perf_counter()
h_idx.add_items(xb, np.arange(n_train, dtype=np.int64))
t_build_hnswlib = time.perf_counter() - t0
print(f"hnswlib build: {t_build_hnswlib:.3f}s")

for ef in [10, 20, 50, 100, 200]:
    h_idx.set_ef(ef)
    labels, _ = h_idx.knn_query(xq, k=10)
    rec = recall_at_k(labels, gt_nbrs, k=10)
    print(f"  hnswlib ef={ef}: recall@10 = {rec:.4f}")

# 3. FAISS
f_idx = faiss.IndexHNSWFlat(dim, 16)
f_idx.hnsw.efConstruction = 100
t0 = time.perf_counter()
f_idx.add(xb)
t_build_faiss = time.perf_counter() - t0
print(f"FAISS build: {t_build_faiss:.3f}s")

for ef in [10, 20, 50, 100, 200]:
    f_idx.hnsw.efSearch = ef
    _, I = f_idx.search(xq, 10)
    rec = recall_at_k(I, gt_nbrs, k=10)
    print(f"  FAISS ef={ef}: recall@10 = {rec:.4f}")
