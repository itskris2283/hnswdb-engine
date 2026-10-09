"""
Microbenchmark for SIMD vs Scalar distance computation kernels.
Measures latency (ns/distance) and throughput (Million distances/sec) across dimensions.
"""

import os
import json
import time
import numpy as np
import hnswdb
from benchmarks.env_info import get_env_info


def benchmark_kernels():
    dims = [16, 32, 64, 128, 784]
    n_pairs = 100000
    results = {}

    env = get_env_info()

    for d in dims:
        rng = np.random.default_rng(42)
        v1 = rng.standard_normal((n_pairs, d)).astype(np.float32)
        v2 = rng.standard_normal((n_pairs, d)).astype(np.float32)

        # Create temporary FlatIndex to exercise distance kernels
        flat_idx = hnswdb.FlatIndex(dim=d, metric="l2")
        flat_idx.add(v1[:1000], np.arange(1000, dtype=np.int64))

        # Warmup
        for _ in range(10):
            flat_idx.search(v2[:100], k=10)

        # Timed search (exercises distance kernel heavily: 1000 queries * 1000 vectors = 1,000,000 distances)
        nq = 1000
        q = v2[:nq]
        t0 = time.perf_counter_ns()
        flat_idx.search(q, k=10)
        t1 = time.perf_counter_ns()

        total_dists = nq * 1000
        elapsed_ns = t1 - t0
        ns_per_dist = elapsed_ns / total_dists
        m_dists_per_sec = (total_dists / (elapsed_ns / 1e9)) / 1e6

        # FLOPs: 3 ops per dimension (diff, square, accumulate)
        gflops = (total_dists * d * 3) / (elapsed_ns)

        results[f"dim_{d}"] = {
            "dim": d,
            "ns_per_distance": round(ns_per_dist, 2),
            "million_distances_per_sec": round(m_dists_per_sec, 2),
            "effective_gflops": round(gflops, 2),
        }
        print(f"Dim {d:3d}: {ns_per_dist:.1f} ns/eval | {m_dists_per_sec:.2f} M-evals/s | {gflops:.2f} GFLOPS")

    out_data = {
        "benchmark": "kernel_microbench",
        "active_kernel": env.get("active_simd_kernel", "unknown"),
        "results": results,
        "env": env,
    }

    out_file = os.path.join("results", "raw", "kernel_microbench.json")
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2)
    print(f"Saved: {out_file}")
    return out_file


if __name__ == "__main__":
    benchmark_kernels()
