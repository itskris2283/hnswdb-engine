#!/usr/bin/env python3
"""
measure_fashion_mnist_self_retrieval.py — Measure corrected self-retrieval rate on Fashion-MNIST (60K, sample 5000 distinct vectors, ef=400).
Outputs results to results/raw/self_retrieval_fashion_mnist_<hash>.json and registers in MANIFEST.json.
"""

import os
import sys
import json
import time
import uuid
import datetime
import subprocess
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import hnswdb
from benchmarks.download_datasets import load_dataset
from benchmarks.env_info import get_env_info

RAW_DIR = os.path.join(ROOT, "results", "raw")
MANIFEST_PATH = os.path.join(RAW_DIR, "MANIFEST.json")


def get_git_commit() -> str:
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except Exception:
        return "unknown"


def main():
    print("Loading Fashion-MNIST dataset...")
    train, _, _, _, metric = load_dataset("fashion-mnist-784-euclidean")
    n_train, dim = train.shape
    M = 16
    ef_c = 100
    ef_s = 400
    sample_size = 5000
    seed = 42

    print(f"Building HNSWDB index (N={n_train}, dim={dim}, M={M}, ef_c={ef_c}, seed={seed})...")
    t0 = time.perf_counter()
    idx = hnswdb.HNSWIndex(dim=dim, metric=metric, max_elements=n_train, M=M, ef_construction=ef_c, seed=seed)
    labels = np.arange(n_train, dtype=np.int64)
    idx.add(train, labels, num_threads=1)
    build_time_s = time.perf_counter() - t0
    print(f"Index built in {build_time_s:.2f} s")

    # Sample 5000 distinct vectors
    rng = np.random.default_rng(seed)
    sample_indices = rng.choice(n_train, size=sample_size, replace=False)
    sample_vecs = np.ascontiguousarray(train[sample_indices], dtype=np.float32)
    sample_labels = labels[sample_indices]

    print(f"Querying {sample_size} vectors with ef_search={ef_s}, k=1...")
    idx.set_ef_search(ef_s)
    t0 = time.perf_counter()
    found_ids, found_dists = idx.search(sample_vecs, k=1, num_threads=1)
    search_time_s = time.perf_counter() - t0
    print(f"Search completed in {search_time_s:.2f} s")

    correct = 0
    unreachable_indices = []
    for i in range(sample_size):
        expected_id = sample_labels[i]
        ret_id = found_ids[i][0]
        dist = found_dists[i][0]
        # Self-retrieval: returned its own label or an identical vector (distance 0.0)
        if ret_id == expected_id or dist <= 1e-5:
            correct += 1
        else:
            unreachable_indices.append(int(expected_id))

    self_retrieval_rate = float(correct / sample_size)
    unreachable_nodes = len(unreachable_indices)
    print(f"Correct: {correct}/{sample_size}")
    print(f"Self-retrieval rate: {self_retrieval_rate:.6f}")
    print(f"Unreachable nodes (in sample): {unreachable_nodes}")

    # Generate run_id and filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    short_hash = uuid.uuid4().hex[:8]
    run_id = f"self_retrieval_fashion_mnist_{timestamp}_{short_hash}"
    filename = f"self_retrieval_fashion_mnist_{short_hash}.json"
    filepath = os.path.join(RAW_DIR, filename)

    result_data = {
        "run_id": run_id,
        "benchmark": "self_retrieval_fashion_mnist",
        "dataset": "fashion_mnist",
        "git_commit": get_git_commit(),
        "timestamp": datetime.datetime.now().isoformat(),
        "params": {
            "M": M,
            "ef_construction": ef_c,
            "ef_search": ef_s,
            "sample_size": sample_size,
            "seed": seed,
            "num_threads": 1,
        },
        "dataset_info": {
            "n_train": n_train,
            "dim": dim,
            "metric": metric,
        },
        "metrics": {
            "self_retrieval_rate": self_retrieval_rate,
            "unreachable_nodes": unreachable_nodes,
            "sample_size": sample_size,
            "build_time_s": round(build_time_s, 4),
            "search_time_s": round(search_time_s, 4),
        },
        "env": get_env_info(),
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=2)
    print(f"Results written to: {filepath}")

    # Register in MANIFEST.json
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    else:
        manifest = {"runs": {}, "last_updated": ""}

    manifest["runs"][run_id] = {
        "file": filename,
        "dataset": "fashion_mnist",
        "benchmark": "self_retrieval_fashion_mnist",
        "timestamp": datetime.datetime.now().isoformat(),
        "git_commit": get_git_commit(),
    }
    manifest["last_updated"] = datetime.datetime.now().isoformat()

    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Registered {run_id} in {MANIFEST_PATH}")

    return result_data


if __name__ == "__main__":
    main()
