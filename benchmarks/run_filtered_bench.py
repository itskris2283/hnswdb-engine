"""
Phase 4: Filtered Search Benchmark.
Evaluates HNSWDB ACORN-style / post-filtering across selectivity thresholds:
1%, 5%, 10%, 25%, 50%, 100%.
Measures recall against exact filtered ground truth, p50 latency, and QPS.
"""

import os
import sys
import time
import json
import uuid
from datetime import datetime, timezone
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import hnswdb
from benchmarks.metrics import (
    recall_at_k,
    measure_single_query_latency,
    measure_batch_qps,
)

RAW_DIR = os.path.join(project_root, "results", "raw")


def run_filtered_benchmark():
    rng = np.random.default_rng(42)
    n_train = 10000
    n_test = 500
    dim = 64
    k = 10

    # Generate data
    xb = rng.standard_normal((n_train, dim)).astype(np.float32)
    xq = rng.standard_normal((n_test, dim)).astype(np.float32)
    ids = np.arange(n_train, dtype=np.int64)

    # Assign metadata categories (integer 0 to 999 for fine-grained selectivity down to 0.1%)
    categories = rng.integers(0, 1000, size=n_train)
    metadata = [{"category": int(cat), "val": int(i)} for i, cat in enumerate(categories)]

    idx = hnswdb.HNSWIndex(dim=dim, metric="l2", max_elements=n_train, M=16, ef_construction=100, seed=42)
    idx.add(xb, ids, metadata=metadata, num_threads=0)

    # Test selectivities: 0.1%, 1%, 5%, 10%, 25%, 50%, 100%
    selectivities = [
        {"name": "0.1pct", "target_pct": 0.1, "filter": {"category": 0}},
        {"name": "1pct", "target_pct": 1.0, "filter": {"category": {"$lt": 10}}},
        {"name": "5pct", "target_pct": 5.0, "filter": {"category": {"$lt": 50}}},
        {"name": "10pct", "target_pct": 10.0, "filter": {"category": {"$lt": 100}}},
        {"name": "25pct", "target_pct": 25.0, "filter": {"category": {"$lt": 250}}},
        {"name": "50pct", "target_pct": 50.0, "filter": {"category": {"$lt": 500}}},
        {"name": "100pct", "target_pct": 100.0, "filter": None},
    ]

    results = []

    for sel in selectivities:
        filt = sel["filter"]
        name = sel["name"]

        # Compute exact brute force ground truth for this filter
        if filt is None:
            matching_mask = np.ones(n_train, dtype=bool)
        elif "category" in filt and isinstance(filt["category"], int):
            matching_mask = (categories == filt["category"])
        elif "category" in filt and "$in" in filt["category"]:
            matching_mask = np.isin(categories, filt["category"]["$in"])
        elif "category" in filt and "$lt" in filt["category"]:
            matching_mask = (categories < filt["category"]["$lt"])
        else:
            matching_mask = np.ones(n_train, dtype=bool)

        actual_matching_count = int(np.sum(matching_mask))
        actual_pct = (actual_matching_count / n_train) * 100.0

        filtered_xb = xb[matching_mask]
        filtered_ids = ids[matching_mask]

        gt_nbrs = np.zeros((n_test, k), dtype=np.int64)
        for qi in range(n_test):
            diff = filtered_xb - xq[qi]
            dists = np.sum(diff * diff, axis=1)
            top_k_sub = np.argsort(dists)[:min(k, len(filtered_xb))]
            gt_nbrs[qi, :len(top_k_sub)] = filtered_ids[top_k_sub]

        # Measure numpy brute force latency
        def bf_search_one(q):
            diff = filtered_xb - q
            dists = np.sum(diff * diff, axis=1)
            return np.argsort(dists)[:min(k, len(filtered_xb))]

        bf_lat = measure_single_query_latency(bf_search_one, xq, warmup=min(50, len(xq)))

        # Benchmark HNSWDB search with this filter
        for ef_s in [50, 100, 200]:
            found_ids, _ = idx.search(xq, k=k, ef_search=ef_s, filter=filt, num_threads=1)
            rec = recall_at_k(found_ids, gt_nbrs, k)

            # Single-query latency
            def search_one(q):
                return idx.search(q, k=k, ef_search=ef_s, filter=filt, num_threads=1)

            lat = measure_single_query_latency(search_one, xq, warmup=min(100, len(xq)))

            # Batch QPS at 1, 4, 8 threads
            qps_threads = {}
            for th in [1, 4, 8]:
                def search_batch(queries, _th=th):
                    return idx.search(queries, k=k, ef_search=ef_s, filter=filt, num_threads=_th)
                qps_threads[f"qps_{th}th"] = measure_batch_qps(search_batch, xq, th, repeats=3)["median"]

            results.append({
                "selectivity_name": name,
                "nominal_pct": sel["target_pct"],
                "actual_matching_count": actual_matching_count,
                "actual_pct": round(actual_pct, 2),
                "ef_search": ef_s,
                "recall_at_10": round(float(rec), 4),
                "p50_ms": round(float(lat["p50"]), 4),
                "p95_ms": round(float(lat["p95"]), 4),
                "p99_ms": round(float(lat["p99"]), 4),
                "sequential_qps": round(float(lat["sequential_qps"]), 1),
                "numpy_bf_p50_ms": round(float(bf_lat["p50"]), 4),
                "numpy_bf_seq_qps": round(float(bf_lat["sequential_qps"]), 1),
                **{k: round(float(v), 1) for k, v in qps_threads.items()},
            })
            print(f"[{name}] ef={ef_s} | R@10={rec:.4f} | p50={lat['p50']:.3f}ms (BF={bf_lat['p50']:.3f}ms) | seqQPS={lat['sequential_qps']:.0f} | 4T_QPS={qps_threads['qps_4th']:.0f}")

    run_id = f"filtered_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    output = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_train": n_train,
        "n_test": n_test,
        "dim": dim,
        "k": k,
        "results": results,
    }

    out_path = os.path.join(RAW_DIR, "filtered_search_benchmark.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nFiltered benchmark saved to: {out_path}")
    return output


if __name__ == "__main__":
    run_filtered_benchmark()
