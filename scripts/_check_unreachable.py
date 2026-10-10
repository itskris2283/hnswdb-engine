"""Quick check: are the unreachable nodes causing the recall failures?"""
import numpy as np
import sys
sys.path.insert(0, ".")

import hnswdb
from benchmarks.datasets_synth import compute_ground_truth_bruteforce

# Generate Dataset B sigma=0.1
rng = np.random.default_rng(42)
n_train, n_test, dim, n_clusters = 20000, 1000, 64, 50
centroids = rng.standard_normal((n_clusters, dim)).astype(np.float32)
train_labels = rng.integers(0, n_clusters, size=n_train)
noise_train = rng.standard_normal((n_train, dim)).astype(np.float32) * 0.1
xb = (centroids[train_labels] + noise_train).astype(np.float32)
test_labels = rng.integers(0, n_clusters, size=n_test)
noise_test = rng.standard_normal((n_test, dim)).astype(np.float32) * 0.1
xq = (centroids[test_labels] + noise_test).astype(np.float32)

gt_nbrs, gt_dists = compute_ground_truth_bruteforce(xb, xq, "l2", 10)

# Build index
idx = hnswdb.HNSWIndex(dim=64, metric="l2", max_elements=20000, M=16, ef_construction=100, seed=42)
idx.add(xb, np.arange(20000, dtype=np.int64), num_threads=1)

# Find query #99 (the 0.0 recall query)
q = np.ascontiguousarray(xq[99:100], dtype=np.float32)
res_ids, res_dists = idx.search(q, k=10, ef_search=200)

print(f"Query #99 search results: {res_ids[0]}")
print(f"Query #99 ground truth:   {gt_nbrs[99]}")
print(f"Intersection: {set(res_ids[0].tolist()) & set(gt_nbrs[99].tolist())}")

# Check which GT neighbors are unreachable
# We can't directly query "unreachable" from Python, but we can check if they show up in search
for gt_id in gt_nbrs[99]:
    # Search for the vector itself
    v = np.ascontiguousarray(xb[gt_id:gt_id+1], dtype=np.float32)
    self_ids, self_dists = idx.search(v, k=1, ef_search=200)
    if self_ids[0][0] != gt_id:
        print(f"  GT neighbor {gt_id} CANNOT be found (self-search returns {self_ids[0][0]} dist={self_dists[0][0]:.6f})")
    else:
        pass

# Also find the 2 unreachable nodes
print("\nSearching for all 20000 nodes via self-retrieval...")
n_unreachable = 0
unreachable_nodes = []
for i in range(20000):
    v = np.ascontiguousarray(xb[i:i+1], dtype=np.float32)
    ids, dists = idx.search(v, k=1, ef_search=500)
    if ids[0][0] != i:
        n_unreachable += 1
        unreachable_nodes.append(i)
        if n_unreachable <= 10:
            print(f"  Node {i} unreachable (self-search returns {ids[0][0]} at dist={dists[0][0]:.8f})")

print(f"\nTotal unreachable: {n_unreachable}")
print(f"Unreachable nodes: {unreachable_nodes[:20]}")

# Check which queries have 0 recall
print("\nQueries with recall < 0.5:")
from benchmarks.metrics import recall_at_k, per_query_recall
q_all = np.ascontiguousarray(xq, dtype=np.float32)
ids_all, _ = idx.search(q_all, k=10, ef_search=200)
pr = per_query_recall(ids_all, gt_nbrs, 10)
bad_queries = np.where(pr < 0.5)[0]
print(f"  {len(bad_queries)} queries with recall < 0.5")

# For each bad query, check if any GT neighbor is unreachable
if unreachable_nodes:
    un_set = set(unreachable_nodes)
    count_un = 0
    for qi in bad_queries:
        gt_set = set(gt_nbrs[qi][:10].tolist())
        if gt_set & un_set:
            count_un += 1
    print(f"  {count_un}/{len(bad_queries)} bad queries have unreachable GT neighbors")
