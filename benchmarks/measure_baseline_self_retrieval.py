#!/usr/bin/env python3
"""
measure_baseline_self_retrieval.py — Measure self-retrieval for hnswlib and FAISS HNSW.

Evaluates:
(a) Dataset B (20K x 64d, 50 clusters, sigma=0.1) for data seeds 0..4, 1-thread build per seed,
    M=16, efConstruction=100, efSearch=400, k=1, 5,000 distinct vectors sampled, distance-zero match.
(b) Fashion-MNIST (60K x 784d, Euclidean) 1-thread build each, M=16, efC=100, efSearch=400, k=1,
    5,000 distinct vectors sampled, distance-zero match.
"""

import os
import sys
import json
import time
import hashlib
import datetime
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import hnswlib
import faiss
import hnswdb
from benchmarks.download_datasets import load_dataset
from benchmarks.run_seed_matrix import generate_dataset_b
from benchmarks.env_info import get_env_info

RAW_DIR = os.path.join(ROOT, "results", "raw")
MANIFEST_PATH = os.path.join(RAW_DIR, "MANIFEST.json")


def evaluate_dataset_b():
    print("Evaluating baseline self-retrieval on Dataset B (seeds 0..4)...")
    sample_size = 5000
    data_seeds = [0, 1, 2, 3, 4]
    M = 16
    ef_c = 100
    ef_s = 400

    results_by_seed = []
    hl_unreach_list = []
    fa_unreach_list = []
    h_unreach_list = []

    hl_rate_list = []
    fa_rate_list = []
    h_rate_list = []

    for d_seed in data_seeds:
        print(f"  Data seed {d_seed}...")
        xb, _, _ = generate_dataset_b(d_seed)
        n, dim = xb.shape
        rng = np.random.default_rng(42)
        sample_idx = rng.choice(n, size=sample_size, replace=False)
        sample_vecs = np.ascontiguousarray(xb[sample_idx], dtype=np.float32)
        sample_labels = np.arange(n, dtype=np.int64)[sample_idx]

        # 1. hnswlib
        p = hnswlib.Index(space="l2", dim=dim)
        p.init_index(max_elements=n, ef_construction=ef_c, M=M, random_seed=42)
        p.add_items(xb, np.arange(n))
        p.set_ef(ef_s)
        hl_labels, hl_dists = p.knn_query(sample_vecs, k=1)
        c_hl = sum(1 for i in range(sample_size) if hl_labels[i][0] == sample_labels[i] or hl_dists[i][0] <= 1e-5)
        unr_hl = sample_size - c_hl
        rate_hl = float(c_hl / sample_size)
        hl_unreach_list.append(unr_hl)
        hl_rate_list.append(rate_hl)

        # 2. faiss
        idx_fa = faiss.IndexHNSWFlat(dim, M)
        idx_fa.hnsw.efConstruction = ef_c
        idx_fa.hnsw.efSearch = ef_s
        idx_fa.add(xb)
        fa_dists, fa_labels = idx_fa.search(sample_vecs, 1)
        c_fa = sum(1 for i in range(sample_size) if fa_labels[i][0] == sample_labels[i] or fa_dists[i][0] <= 1e-5)
        unr_fa = sample_size - c_fa
        rate_fa = float(c_fa / sample_size)
        fa_unreach_list.append(unr_fa)
        fa_rate_list.append(rate_fa)

        # 3. hnswdb cured
        opt_cured = hnswdb.BuildOptions("M", False, False, "heuristic", "closest")
        idx_h = hnswdb.HNSWIndex(dim=dim, metric="l2", max_elements=n, M=M, ef_construction=ef_c, seed=42, build_options=opt_cured)
        idx_h.add(xb, np.arange(n, dtype=np.int64), num_threads=1)
        idx_h.set_ef_search(ef_s)
        h_labels, h_dists = idx_h.search(sample_vecs, k=1, num_threads=1)
        c_h = sum(1 for i in range(sample_size) if h_labels[i][0] == sample_labels[i] or h_dists[i][0] <= 1e-5)
        unr_h = sample_size - c_h
        rate_h = float(c_h / sample_size)
        h_unreach_list.append(unr_h)
        h_rate_list.append(rate_h)

        results_by_seed.append({
            "data_seed": d_seed,
            "hnswlib": {"unreachable_in_sample": unr_hl, "self_retrieval_rate": rate_hl},
            "faiss": {"unreachable_in_sample": unr_fa, "self_retrieval_rate": rate_fa},
            "hnswdb": {"unreachable_in_sample": unr_h, "self_retrieval_rate": rate_h},
        })

    summary = {
        "dataset": "Dataset B (20K x 64d, sigma=0.1)",
        "sample_size": sample_size,
        "ef_search": ef_s,
        "seeds": data_seeds,
        "results_by_seed": results_by_seed,
        "hnswlib": {
            "unreachable_mean": float(np.mean(hl_unreach_list)),
            "self_retrieval_rate_mean": float(np.mean(hl_rate_list)),
            "unreachable_counts": hl_unreach_list,
        },
        "faiss": {
            "unreachable_mean": float(np.mean(fa_unreach_list)),
            "self_retrieval_rate_mean": float(np.mean(fa_rate_list)),
            "unreachable_counts": fa_unreach_list,
        },
        "hnswdb": {
            "unreachable_mean": float(np.mean(h_unreach_list)),
            "self_retrieval_rate_mean": float(np.mean(h_rate_list)),
            "unreachable_counts": h_unreach_list,
        },
    }
    return summary


def evaluate_fashion_mnist():
    print("Evaluating baseline self-retrieval on Fashion-MNIST...")
    train, _, _, _, _ = load_dataset("fashion-mnist-784-euclidean")
    n_train, dim = train.shape
    sample_size = 5000
    M = 16
    ef_c = 100
    ef_s = 400

    rng = np.random.default_rng(42)
    sample_idx = rng.choice(n_train, size=sample_size, replace=False)
    sample_vecs = np.ascontiguousarray(train[sample_idx], dtype=np.float32)
    sample_labels = np.arange(n_train, dtype=np.int64)[sample_idx]

    # 1. hnswlib
    print("  hnswlib...")
    p = hnswlib.Index(space="l2", dim=dim)
    p.init_index(max_elements=n_train, ef_construction=ef_c, M=M, random_seed=42)
    p.add_items(train, np.arange(n_train))
    p.set_ef(ef_s)
    hl_labels, hl_dists = p.knn_query(sample_vecs, k=1)
    c_hl = sum(1 for i in range(sample_size) if hl_labels[i][0] == sample_labels[i] or hl_dists[i][0] <= 1e-5)
    unr_hl = sample_size - c_hl
    rate_hl = float(c_hl / sample_size)

    # 2. faiss
    print("  faiss...")
    idx_fa = faiss.IndexHNSWFlat(dim, M)
    idx_fa.hnsw.efConstruction = ef_c
    idx_fa.hnsw.efSearch = ef_s
    idx_fa.add(train)
    fa_dists, fa_labels = idx_fa.search(sample_vecs, 1)
    c_fa = sum(1 for i in range(sample_size) if fa_labels[i][0] == sample_labels[i] or fa_dists[i][0] <= 1e-5)
    unr_fa = sample_size - c_fa
    rate_fa = float(c_fa / sample_size)

    # 3. hnswdb cured
    print("  hnswdb...")
    opt_cured = hnswdb.BuildOptions("M", False, False, "heuristic", "closest")
    idx_h = hnswdb.HNSWIndex(dim=dim, metric="l2", max_elements=n_train, M=M, ef_construction=ef_c, seed=42, build_options=opt_cured)
    idx_h.add(train, np.arange(n_train, dtype=np.int64), num_threads=1)
    idx_h.set_ef_search(ef_s)
    h_labels, h_dists = idx_h.search(sample_vecs, k=1, num_threads=1)
    c_h = sum(1 for i in range(sample_size) if h_labels[i][0] == sample_labels[i] or h_dists[i][0] <= 1e-5)
    unr_h = sample_size - c_h
    rate_h = float(c_h / sample_size)

    summary = {
        "dataset": "fashion_mnist",
        "sample_size": sample_size,
        "ef_search": ef_s,
        "hnswlib": {
            "unreachable_count": unr_hl,
            "self_retrieval_rate": rate_hl,
        },
        "faiss": {
            "unreachable_count": unr_fa,
            "self_retrieval_rate": rate_fa,
        },
        "hnswdb": {
            "unreachable_count": unr_h,
            "self_retrieval_rate": rate_h,
        },
    }
    return summary


def main():
    db_summary = evaluate_dataset_b()
    fm_summary = evaluate_fashion_mnist()

    payload = {
        "benchmark": "baseline_self_retrieval",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "params": {"M": 16, "ef_construction": 100, "ef_search": 400, "sample_size": 5000},
        "dataset_b": db_summary,
        "fashion_mnist": fm_summary,
        "env": get_env_info(),
    }

    # Deterministic content hash for run_id
    payload_bytes = json.dumps(payload, sort_keys=True).encode("utf-8")
    content_hash = hashlib.sha256(payload_bytes).hexdigest()[:8]
    run_id = f"baseline_self_retrieval_{content_hash}"
    payload["run_id"] = run_id

    filename = f"baseline_self_retrieval_{content_hash}.json"
    filepath = os.path.join(RAW_DIR, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved results to: {filepath} (run_id={run_id})")

    # Register in MANIFEST.json
    with open(filepath, "rb") as f:
        data = f.read()
    sha256 = hashlib.sha256(data).hexdigest()
    size_bytes = len(data)

    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, "r", encoding="utf-8") as mf:
            manifest = json.load(mf)
        runs = manifest.get("runs", [])
        # Append entry
        runs.append({
            "file": filename,
            "run_id": run_id,
            "dataset": "dataset_b, fashion_mnist",
            "label": "Baseline Self-Retrieval (hnswlib and FAISS, 5K sample, ef=400)",
            "sha256": sha256,
            "size_bytes": size_bytes,
        })
        manifest["runs"] = runs
        with open(MANIFEST_PATH, "w", encoding="utf-8") as mf:
            json.dump(manifest, mf, indent=2)
        print(f"Registered {filename} in MANIFEST.json")

    return filepath


if __name__ == "__main__":
    main()
