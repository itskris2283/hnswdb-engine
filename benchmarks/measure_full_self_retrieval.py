#!/usr/bin/env python3
"""
measure_full_self_retrieval.py — Measure full-index self-retrieval for Dataset B and Fashion-MNIST.

Ensures identical denominators across all engines:
- Dataset B: Full index (20,000 nodes, 64d, sigma=0.1) across data seeds 0..4 (single-thread build, M=16, efC=100, efSearch=400).
- Fashion-MNIST: Full index (60,000 nodes, 784d) for hnswlib, FAISS, and HNSWDB (M=16, efC=100, efSearch=400).
  Also measures 5,000 distinct vector sample under identical indices for direct sample comparison.
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


def run_benchmark():
    print("=== Starting Full-Index Self-Retrieval Measurement ===")

    # 1. Dataset B (20,000 nodes per seed, seeds 0..4)
    print("\n--- Measuring Dataset B (20,000 nodes, seeds 0..4) ---")
    data_seeds = [0, 1, 2, 3, 4]
    denom_b = 20000
    M = 16
    ef_c = 100
    ef_s = 400

    db_results_by_seed = []
    hl_unreach_b = []
    fa_unreach_b = []
    h_unreach_b = []

    for d_seed in data_seeds:
        print(f"  Dataset B data seed {d_seed}...")
        xb, _, _ = generate_dataset_b(d_seed)
        n, dim = xb.shape
        assert n == denom_b
        labels = np.arange(n, dtype=np.int64)

        # hnswlib
        p = hnswlib.Index(space="l2", dim=dim)
        p.init_index(max_elements=n, ef_construction=ef_c, M=M, random_seed=42)
        p.add_items(xb, labels, num_threads=1)
        p.set_ef(ef_s)
        hl_labels, hl_dists = p.knn_query(xb, k=1, num_threads=1)
        c_hl = sum(1 for i in range(n) if hl_labels[i][0] == labels[i] or hl_dists[i][0] <= 1e-5)
        u_hl = n - c_hl
        r_hl = float(c_hl / n)
        hl_unreach_b.append(u_hl)

        # FAISS
        idx_fa = faiss.IndexHNSWFlat(dim, M)
        idx_fa.hnsw.efConstruction = ef_c
        idx_fa.hnsw.efSearch = ef_s
        idx_fa.add(xb)
        fa_dists, fa_labels = idx_fa.search(xb, 1)
        c_fa = sum(1 for i in range(n) if fa_labels[i][0] == labels[i] or fa_dists[i][0] <= 1e-5)
        u_fa = n - c_fa
        r_fa = float(c_fa / n)
        fa_unreach_b.append(u_fa)

        # HNSWDB cured
        opt_cured = hnswdb.BuildOptions("M", False, False, "heuristic", "closest")
        idx_h = hnswdb.HNSWIndex(dim=dim, metric="l2", max_elements=n, M=M, ef_construction=ef_c, seed=42, build_options=opt_cured)
        idx_h.add(xb, labels, num_threads=1)
        idx_h.set_ef_search(ef_s)
        h_labels, h_dists = idx_h.search(xb, k=1, num_threads=1)
        c_h = sum(1 for i in range(n) if h_labels[i][0] == labels[i] or h_dists[i][0] <= 1e-5)
        u_h = n - c_h
        r_h = float(c_h / n)
        h_unreach_b.append(u_h)

        db_results_by_seed.append({
            "data_seed": d_seed,
            "denominator": denom_b,
            "hnswlib": {"unreachable_count": u_hl, "failure_rate": float(u_hl / n), "self_retrieval_rate": r_hl},
            "faiss": {"unreachable_count": u_fa, "failure_rate": float(u_fa / n), "self_retrieval_rate": r_fa},
            "hnswdb": {"unreachable_count": u_h, "failure_rate": float(u_h / n), "self_retrieval_rate": r_h},
        })

    dataset_b_summary = {
        "dataset": "Dataset B (20K x 64d, sigma=0.1)",
        "denominator": denom_b,
        "sample_size": denom_b,
        "n_builds": len(data_seeds),
        "seeds": data_seeds,
        "ef_search": ef_s,
        "results_by_seed": db_results_by_seed,
        "hnswlib": {
            "unreachable_mean": float(np.mean(hl_unreach_b)),
            "unreachable_counts": hl_unreach_b,
            "denominator": denom_b,
            "failure_rate_mean": float(np.mean(hl_unreach_b) / denom_b),
            "self_retrieval_rate_mean": float(1.0 - np.mean(hl_unreach_b) / denom_b),
            "n_builds": len(data_seeds),
        },
        "faiss": {
            "unreachable_mean": float(np.mean(fa_unreach_b)),
            "unreachable_counts": fa_unreach_b,
            "denominator": denom_b,
            "failure_rate_mean": float(np.mean(fa_unreach_b) / denom_b),
            "self_retrieval_rate_mean": float(1.0 - np.mean(fa_unreach_b) / denom_b),
            "n_builds": len(data_seeds),
        },
        "hnswdb": {
            "unreachable_mean": float(np.mean(h_unreach_b)),
            "unreachable_counts": h_unreach_b,
            "denominator": denom_b,
            "failure_rate_mean": float(np.mean(h_unreach_b) / denom_b),
            "self_retrieval_rate_mean": float(1.0 - np.mean(h_unreach_b) / denom_b),
            "n_builds": len(data_seeds),
        },
    }

    # 2. Fashion-MNIST (60,000 nodes full index & 5,000 sample)
    print("\n--- Measuring Fashion-MNIST (60,000 nodes full index) ---")
    train, _, _, _, _ = load_dataset("fashion-mnist-784-euclidean")
    n_train, dim_fm = train.shape
    denom_fm = n_train
    labels_fm = np.arange(n_train, dtype=np.int64)

    # 2a. Full Index 60,000
    # hnswlib
    print("  Fashion-MNIST hnswlib (full index)...")
    p_fm = hnswlib.Index(space="l2", dim=dim_fm)
    p_fm.init_index(max_elements=n_train, ef_construction=ef_c, M=M, random_seed=42)
    p_fm.add_items(train, labels_fm, num_threads=1)
    p_fm.set_ef(ef_s)
    hl_labels_fm, hl_dists_fm = p_fm.knn_query(train, k=1, num_threads=1)
    c_hl_fm = sum(1 for i in range(n_train) if hl_labels_fm[i][0] == labels_fm[i] or hl_dists_fm[i][0] <= 1e-5)
    u_hl_fm = n_train - c_hl_fm
    r_hl_fm = float(c_hl_fm / n_train)

    # FAISS
    print("  Fashion-MNIST FAISS (full index)...")
    idx_fa_fm = faiss.IndexHNSWFlat(dim_fm, M)
    idx_fa_fm.hnsw.efConstruction = ef_c
    idx_fa_fm.hnsw.efSearch = ef_s
    idx_fa_fm.add(train)
    fa_dists_fm, fa_labels_fm = idx_fa_fm.search(train, 1)
    c_fa_fm = sum(1 for i in range(n_train) if fa_labels_fm[i][0] == labels_fm[i] or fa_dists_fm[i][0] <= 1e-5)
    u_fa_fm = n_train - c_fa_fm
    r_fa_fm = float(c_fa_fm / n_train)

    # HNSWDB cured
    print("  Fashion-MNIST HNSWDB (full index)...")
    idx_h_fm = hnswdb.HNSWIndex(dim=dim_fm, metric="l2", max_elements=n_train, M=M, ef_construction=ef_c, seed=42, build_options=opt_cured)
    idx_h_fm.add(train, labels_fm, num_threads=1)
    idx_h_fm.set_ef_search(ef_s)
    h_labels_fm, h_dists_fm = idx_h_fm.search(train, k=1, num_threads=1)
    c_h_fm = sum(1 for i in range(n_train) if h_labels_fm[i][0] == labels_fm[i] or h_dists_fm[i][0] <= 1e-5)
    u_h_fm = n_train - c_h_fm
    r_h_fm = float(c_h_fm / n_train)

    # 2b. Also extract sample of 5,000 distinct vectors (same seed 42)
    rng_sample = np.random.default_rng(42)
    sample_indices = rng_sample.choice(n_train, size=5000, replace=False)
    sample_labels = labels_fm[sample_indices]

    c_hl_samp = sum(1 for idx_pos, orig_idx in enumerate(sample_indices)
                    if hl_labels_fm[orig_idx][0] == sample_labels[idx_pos] or hl_dists_fm[orig_idx][0] <= 1e-5)
    c_fa_samp = sum(1 for idx_pos, orig_idx in enumerate(sample_indices)
                    if fa_labels_fm[orig_idx][0] == sample_labels[idx_pos] or fa_dists_fm[orig_idx][0] <= 1e-5)
    c_h_samp = sum(1 for idx_pos, orig_idx in enumerate(sample_indices)
                   if h_labels_fm[orig_idx][0] == sample_labels[idx_pos] or h_dists_fm[orig_idx][0] <= 1e-5)

    fashion_mnist_summary = {
        "dataset": "fashion_mnist",
        "denominator": denom_fm,
        "sample_size": denom_fm,
        "n_builds": 1,
        "ef_search": ef_s,
        "full_index_60000": {
            "denominator": denom_fm,
            "hnswlib": {
                "unreachable_count": u_hl_fm,
                "failure_rate": float(u_hl_fm / denom_fm),
                "self_retrieval_rate": r_hl_fm,
                "n_builds": 1,
            },
            "faiss": {
                "unreachable_count": u_fa_fm,
                "failure_rate": float(u_fa_fm / denom_fm),
                "self_retrieval_rate": r_fa_fm,
                "n_builds": 1,
            },
            "hnswdb": {
                "unreachable_count": u_h_fm,
                "failure_rate": float(u_h_fm / denom_fm),
                "self_retrieval_rate": r_h_fm,
                "n_builds": 1,
            },
        },
        "sample_5000": {
            "denominator": 5000,
            "hnswlib": {
                "unreachable_count": 5000 - c_hl_samp,
                "failure_rate": float((5000 - c_hl_samp) / 5000),
                "self_retrieval_rate": float(c_hl_samp / 5000),
                "n_builds": 1,
            },
            "faiss": {
                "unreachable_count": 5000 - c_fa_samp,
                "failure_rate": float((5000 - c_fa_samp) / 5000),
                "self_retrieval_rate": float(c_fa_samp / 5000),
                "n_builds": 1,
            },
            "hnswdb": {
                "unreachable_count": 5000 - c_h_samp,
                "failure_rate": float((5000 - c_h_samp) / 5000),
                "self_retrieval_rate": float(c_h_samp / 5000),
                "n_builds": 1,
            },
        },
        # Primary reference uses full index (60,000 nodes)
        "hnswlib": {
            "unreachable_count": u_hl_fm,
            "denominator": denom_fm,
            "failure_rate": float(u_hl_fm / denom_fm),
            "self_retrieval_rate": r_hl_fm,
            "n_builds": 1,
        },
        "faiss": {
            "unreachable_count": u_fa_fm,
            "denominator": denom_fm,
            "failure_rate": float(u_fa_fm / denom_fm),
            "self_retrieval_rate": r_fa_fm,
            "n_builds": 1,
        },
        "hnswdb": {
            "unreachable_count": u_h_fm,
            "denominator": denom_fm,
            "failure_rate": float(u_h_fm / denom_fm),
            "self_retrieval_rate": r_h_fm,
            "n_builds": 1,
        },
    }

    env = get_env_info()

    data_to_hash = json.dumps({"dataset_b": dataset_b_summary, "fashion_mnist": fashion_mnist_summary}, sort_keys=True)
    content_hash = hashlib.sha256(data_to_hash.encode("utf-8")).hexdigest()[:8]
    run_id = f"consistent_self_retrieval_{content_hash}"

    full_output = {
        "benchmark": "consistent_self_retrieval",
        "run_id": run_id,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "params": {
            "M": M,
            "ef_construction": ef_c,
            "ef_search": ef_s,
            "dataset_b_denominator": denom_b,
            "fashion_mnist_denominator": denom_fm,
            "rule": "distance_zero_or_label_match",
        },
        "dataset_b": dataset_b_summary,
        "fashion_mnist": fashion_mnist_summary,
        "env": env,
    }

    out_filename = f"consistent_self_retrieval_{content_hash}.json"
    out_path = os.path.join(RAW_DIR, out_filename)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2)

    with open(out_path, "rb") as f:
        sha256_hash = hashlib.sha256(f.read()).hexdigest()
    size_bytes = os.path.getsize(out_path)

    print(f"\nSaved raw results to: {out_path}")
    print(f"run_id: {run_id}")
    print(f"sha256: {sha256_hash}")
    print(f"size_bytes: {size_bytes}")

    # Register in MANIFEST.json
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, "r", encoding="utf-8") as mf:
            manifest = json.load(mf)

        # Remove previous entry for same file if present
        manifest["runs"] = [r for r in manifest["runs"] if r["file"] != out_filename]
        manifest["runs"].append({
            "file": out_filename,
            "run_id": run_id,
            "dataset": "dataset_b, fashion_mnist",
            "label": "Consistent Full-Index Self-Retrieval (Dataset B 20K, Fashion-MNIST 60K)",
            "sha256": sha256_hash,
            "size_bytes": size_bytes,
        })
        manifest["last_updated"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        with open(MANIFEST_PATH, "w", encoding="utf-8") as mf:
            json.dump(manifest, mf, indent=2)
        print(f"Registered {out_filename} in MANIFEST.json")

    return out_filename, run_id


if __name__ == "__main__":
    run_benchmark()
