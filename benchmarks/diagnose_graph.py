import numpy as np
import hnswdb
from benchmarks.download_datasets import load_dataset


def diagnose_hnsw_index(dim=32, n_elements=10000, M=16, ef_c=100, metric="l2", seed=42):
    train_data, _, _, _, _ = load_dataset(f"synthetic_{n_elements}_{dim}_{metric}")
    index = hnswdb.HNSWIndex(dim=dim, metric=metric, max_elements=n_elements, M=M, ef_construction=ef_c, seed=seed)
    ids = np.arange(n_elements, dtype=np.int64)
    index.add(train_data, ids)

    stats = index.stats()
    max_level = int(stats.get("max_level", 0))
    entry_point = int(stats.get("entry_point", 0))

    print(f"--- Graph Diagnostics (N={n_elements}, M={M}, ef_c={ef_c}) ---")
    print(f"Max Level: {max_level}, Entry Point: {entry_point}")

    # Check connectivity and reachability using queries
    test_queries = train_data[:100]
    recalls = []
    for ef_s in [10, 20, 50, 100, 200, 500]:
        found_ids, found_dists = index.search(test_queries, k=1, ef_search=ef_s)
        # For training queries, k=1 must be exact match (the vector itself, dist == 0)
        exact_matches = sum(found_ids[i, 0] == i for i in range(len(test_queries)))
        recalls.append((ef_s, exact_matches / len(test_queries)))
        print(f"  Self-recall@1 at ef_search={ef_s:3d}: {exact_matches}/{len(test_queries)} ({exact_matches/len(test_queries):.4f})")

    return stats


if __name__ == "__main__":
    diagnose_hnsw_index()
