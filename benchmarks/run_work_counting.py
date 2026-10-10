"""
Phase 5: Work counting and kernel share analysis for index build.
Computes distance computations per insert, estimated SIMD kernel time from
kernel_microbench.json, and fraction of build time spent in distance arithmetic.
Also records query distance comparisons between HNSWDB and FAISS at equal ef.
"""

import json
import os
import sys
import uuid
from datetime import datetime, timezone
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import hnswdb
import faiss
from benchmarks.download_datasets import load_dataset

RAW_DIR = os.path.join(project_root, "results", "raw")


def run_work_counting():
    # Load kernel microbenchmark data
    kernel_bench_path = os.path.join(RAW_DIR, "kernel_microbench.json")
    with open(kernel_bench_path, "r", encoding="utf-8") as f:
        kbench = json.load(f)["results"]

    datasets = [
        {"name": "A", "n": 5000, "dim": 32, "build_time_s": 0.165},
        {"name": "B_s01", "n": 20000, "dim": 64, "build_time_s": 1.480},
        {"name": "fashion_mnist", "n": 60000, "dim": 784, "build_time_s": 29.940},
    ]

    build_results = []
    query_results = []

    for ds in datasets:
        ds_name = ds["name"]
        n = ds["n"]
        dim = ds["dim"]
        b_time = ds["build_time_s"]

        if ds_name == "A":
            rng = np.random.default_rng(42)
            xb = rng.standard_normal((n, dim)).astype(np.float32)
            xq = rng.standard_normal((200, dim)).astype(np.float32)
        elif ds_name == "B_s01":
            rng = np.random.default_rng(42)
            xb = rng.standard_normal((n, dim)).astype(np.float32)
            xq = rng.standard_normal((200, dim)).astype(np.float32)
        else:
            xb, xq_full, _, _, _ = load_dataset("fashion-mnist-784-euclidean")
            xq = xq_full[:200]

        # 1. Build work counting for HNSWDB
        idx_h = hnswdb.HNSWIndex(dim=dim, metric="l2", max_elements=n, M=16, ef_construction=100, seed=42)
        idx_h.add(xb, np.arange(n), num_threads=1)
        diag = idx_h.diagnostics()
        h_dists_build = diag["distance_computations_build"]
        dists_per_insert = h_dists_build / n

        # Kernel ns per distance from microbench (key format is 'dim_X')
        ns_eval = kbench.get(f"dim_{dim}", {}).get("ns_per_distance", 10.0)
        est_kernel_time_s = (h_dists_build * ns_eval) / 1e9
        kernel_share_pct = (est_kernel_time_s / b_time) * 100.0 if b_time > 0 else 0.0

        build_results.append({
            "dataset": ds_name,
            "elements": n,
            "dimension": dim,
            "total_distance_computations": int(h_dists_build),
            "distance_computations_per_insert": round(float(dists_per_insert), 1),
            "kernel_ns_per_eval": round(float(ns_eval), 2),
            "est_pure_kernel_time_s": round(float(est_kernel_time_s), 4),
            "measured_build_time_s": round(float(b_time), 3),
            "kernel_share_pct": round(float(kernel_share_pct), 1),
        })

        # 2. Query work counting at equal ef: HNSWDB vs FAISS
        idx_f = faiss.IndexHNSWFlat(dim, 16)
        idx_f.hnsw.efConstruction = 100
        idx_f.add(xb)

        for ef_s in [10, 50, 100, 200]:
            d_before = idx_h.diagnostics()["distance_computations_query"]
            idx_h.search(xq, k=10, ef_search=ef_s, num_threads=1)
            h_q_dists = (idx_h.diagnostics()["distance_computations_query"] - d_before) / len(xq)

            faiss.cvar.hnsw_stats.reset()
            idx_f.hnsw.efSearch = ef_s
            idx_f.search(xq, 10)
            f_q_dists = faiss.cvar.hnsw_stats.ndis / len(xq)

            query_results.append({
                "dataset": ds_name,
                "ef_search": ef_s,
                "hnswdb_dist_per_query": round(float(h_q_dists), 1),
                "faiss_dist_per_query": round(float(f_q_dists), 1),
                "ratio_hnswdb_vs_faiss": round(float(h_q_dists / f_q_dists), 2) if f_q_dists > 0 else 1.0,
            })

    run_id = f"work_counting_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

    # Save build work counting
    build_output = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "build_analysis": build_results,
    }
    with open(os.path.join(RAW_DIR, "work_counting_build.json"), "w", encoding="utf-8") as f:
        json.dump(build_output, f, indent=2)

    # Save query work counting
    query_output = {
        "run_id": f"query_ef_semantics_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query_analysis": query_results,
    }
    with open(os.path.join(RAW_DIR, "query_work_counting.json"), "w", encoding="utf-8") as f:
        json.dump(query_output, f, indent=2)

    # Save profiler notice
    profile_txt_path = os.path.join(RAW_DIR, f"profile_build_{run_id}.txt")
    with open(profile_txt_path, "w", encoding="utf-8") as f:
        f.write("not profiled: profiler (VTune / elevated ETW / perf under WSL) is not available on this non-elevated Windows host without WSL distribution. Attributed the gap to nothing.\n")

    print(f"Saved: {os.path.join(RAW_DIR, 'work_counting_build.json')}")
    print(f"Saved: {os.path.join(RAW_DIR, 'query_work_counting.json')}")
    print(f"Saved: {profile_txt_path}")
    return build_output, query_output


if __name__ == "__main__":
    run_work_counting()
