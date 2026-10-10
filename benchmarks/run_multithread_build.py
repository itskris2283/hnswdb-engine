"""
Run multi-threaded index construction benchmark at max physical threads (4 threads)
for HNSWDB vs hnswlib on Dataset B and Fashion-MNIST.
Outputs results/raw/multithread_build.json to evaluate Target T9.
"""

import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import hnswdb
import hnswlib
from benchmarks.download_datasets import load_dataset

RAW_DIR = os.path.join(project_root, "results", "raw")


def run_benchmark():
    num_threads = 4  # max physical threads on this host
    repeats = 3

    # 1. Dataset B (20K x 64d)
    rng = np.random.default_rng(42)
    n_b = 20000
    dim_b = 64
    xb_b = rng.standard_normal((n_b, dim_b)).astype(np.float32)
    ids_b = np.arange(n_b, dtype=np.int64)

    h_b_times = []
    hl_b_times = []
    for rep in range(repeats):
        # HNSWDB
        t0 = time.perf_counter()
        idx_h = hnswdb.HNSWIndex(dim=dim_b, metric="l2", max_elements=n_b, M=16, ef_construction=100, seed=42 + rep)
        idx_h.add(xb_b, ids_b, num_threads=num_threads)
        h_b_times.append(time.perf_counter() - t0)

        # hnswlib
        t0 = time.perf_counter()
        idx_hl = hnswlib.Index(space="l2", dim=dim_b)
        idx_hl.init_index(max_elements=n_b, ef_construction=100, M=16, random_seed=42 + rep)
        idx_hl.add_items(xb_b, ids_b, num_threads=num_threads)
        hl_b_times.append(time.perf_counter() - t0)

    med_h_b = float(np.median(h_b_times))
    med_hl_b = float(np.median(hl_b_times))
    ratio_b = med_h_b / med_hl_b

    print(f"Dataset B (4T): HNSWDB={med_h_b:.3f}s | hnswlib={med_hl_b:.3f}s | ratio={ratio_b:.2f}x")

    # 2. Fashion-MNIST (60K x 784d)
    xb_f, _, _, _, _ = load_dataset("fashion-mnist-784-euclidean")
    ids_f = np.arange(len(xb_f), dtype=np.int64)
    dim_f = xb_f.shape[1]

    h_f_times = []
    hl_f_times = []
    for rep in range(repeats):
        # HNSWDB
        t0 = time.perf_counter()
        idx_h = hnswdb.HNSWIndex(dim=dim_f, metric="l2", max_elements=len(xb_f), M=16, ef_construction=100, seed=42 + rep)
        idx_h.add(xb_f, ids_f, num_threads=num_threads)
        h_f_times.append(time.perf_counter() - t0)

        # hnswlib
        t0 = time.perf_counter()
        idx_hl = hnswlib.Index(space="l2", dim=dim_f)
        idx_hl.init_index(max_elements=len(xb_f), ef_construction=100, M=16, random_seed=42 + rep)
        idx_hl.add_items(xb_f, ids_f, num_threads=num_threads)
        hl_f_times.append(time.perf_counter() - t0)

    med_h_f = float(np.median(h_f_times))
    med_hl_f = float(np.median(hl_f_times))
    ratio_f = med_h_f / med_hl_f

    print(f"Fashion-MNIST (4T): HNSWDB={med_h_f:.3f}s | hnswlib={med_hl_f:.3f}s | ratio={ratio_f:.2f}x")

    max_ratio = max(ratio_b, ratio_f)
    print(f"Max ratio at 4T: {max_ratio:.2f}x (target <= 3.0x: {'PASS' if max_ratio <= 3.0 else 'MISS'})")

    run_id = f"multithread_build_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    output = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "num_threads": num_threads,
        "repeats": repeats,
        "dataset_b": {
            "hnswdb_build_s": round(med_h_b, 3),
            "hnswlib_build_s": round(med_hl_b, 3),
            "ratio": round(ratio_b, 2),
        },
        "fashion_mnist": {
            "hnswdb_build_s": round(med_h_f, 3),
            "hnswlib_build_s": round(med_hl_f, 3),
            "ratio": round(ratio_f, 2),
        },
        "max_threads_ratio_hnswlib": round(max_ratio, 2),
    }

    out_file = os.path.join(RAW_DIR, "multithread_build.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"Saved: {out_file}")
    return output


if __name__ == "__main__":
    run_benchmark()
