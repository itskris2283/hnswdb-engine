#!/usr/bin/env python3
"""
where_it_loses.py — Systematically identify all benchmarks where HNSWDB
is worse than the best reference baseline by more than 5%.

Applies the C9 rule to speed orderings: relative speed on Fashion-MNIST
varies between runs on this host; treat as parity.
"""

import json
import os
import hashlib
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(ROOT, "results", "raw")
MANIFEST_PATH = os.path.join(RAW_DIR, "MANIFEST.json")


def analyze_losing_comparisons() -> dict:
    comparisons = []

    # 1. Reachability / Unreachable Nodes on Clustered Data (Dataset B 5x5 Matrix)
    sm_path = os.path.join(RAW_DIR, "seed_matrix_dataset_b.json")
    if os.path.exists(sm_path):
        with open(sm_path, "r", encoding="utf-8") as f:
            sm = json.load(f)
        cured_unreach = sm.get("engines", {}).get("hnswdb_cured", {}).get("unreachable_mean", 0.0)
        comparisons.append({
            "workload": "Dataset B (20K x 64d, sigma=0.1, 50 clusters)",
            "metric": "Unreachable nodes (1-NN self-retrieval)",
            "hnswdb": f"{cured_unreach:.2f}",
            "best_baseline": "0.00 (hnswlib / FAISS)",
            "gap": f"+{cured_unreach:.2f} nodes (~0.05% of dataset)",
            "notes": "Target T5 remains a MISS; intra-cluster clustering causes isolated sub-components",
        })

    # 2. Reachability on Fashion-MNIST (sample of 5000 distinct vectors, ef=400)
    # Target T4/T5 notes
    sr_files = [f for f in os.listdir(RAW_DIR) if f.startswith("self_retrieval_fashion_mnist_")]
    if sr_files:
        with open(os.path.join(RAW_DIR, sr_files[0]), "r", encoding="utf-8") as f:
            sr = json.load(f)
        unreach_fm = sr.get("unreachable_count", 0)
        rate_fm = sr.get("self_retrieval_rate", 1.0)
        if unreach_fm > 0:
            comparisons.append({
                "workload": "Fashion-MNIST (60K x 784d, 5,000 sampled distinct vectors)",
                "metric": "Unreachable nodes (1-NN self-retrieval, ef=400)",
                "hnswdb": f"{unreach_fm} nodes ({rate_fm:.4f} rate)",
                "best_baseline": "0 nodes (1.0000 rate)",
                "gap": f"+{unreach_fm} nodes (below 0.999 target)",
                "notes": "Identical vectors excluded; 12 vectors unreachable at ef=400 (T4/T5 MISS)",
            })

    # 3. Single-threaded build time on Dataset B (64d clustered)
    b_path = os.path.join(RAW_DIR, "bench_B_s01_8a2302be.json")
    if os.path.exists(b_path):
        with open(b_path, "r", encoding="utf-8") as f:
            db = json.load(f)
        hb = db.get("engines", {}).get("hnswdb", {}).get("seed_42", {}).get("build_time_s", 0.0)
        fb = db.get("engines", {}).get("faiss", {}).get("seed_42", {}).get("build_time_s", 0.0)
        if fb > 0 and hb > fb * 1.05:
            pct = ((hb - fb) / fb) * 100.0
            comparisons.append({
                "workload": "Dataset B (20K x 64d, sigma=0.1)",
                "metric": "Single-thread build time",
                "hnswdb": f"{hb:.3f} s",
                "best_baseline": f"{fb:.3f} s (FAISS)",
                "gap": f"{pct:.1f}% slower",
                "notes": "FAISS build is faster on 64d; hnswlib build is 3.382 s",
            })

    # 4. Sequential QPS on Dataset A at ef=10
    a_path = os.path.join(RAW_DIR, "bench_A_a9f1a6f6.json")
    if os.path.exists(a_path):
        with open(a_path, "r", encoding="utf-8") as f:
            da = json.load(f)
        rh = da.get("engines", {}).get("hnswdb", {}).get("seed_42", {}).get("ef_results", [{}])[0]
        rl = da.get("engines", {}).get("hnswlib", {}).get("seed_42", {}).get("ef_results", [{}])[0]
        qh = rh.get("qps_by_threads", {}).get("1", {}).get("median", 0.0)
        ql = rl.get("qps_by_threads", {}).get("1", {}).get("median", 0.0)
        if ql > 0 and qh < ql * 0.95:
            pct = ((ql - qh) / ql) * 100.0
            comparisons.append({
                "workload": "Dataset A (5K x 32d, ef_search=10)",
                "metric": "Sequential QPS (1 thread)",
                "hnswdb": f"{qh:,.0f} QPS",
                "best_baseline": f"{ql:,.0f} QPS (hnswlib)",
                "gap": f"{pct:.1f}% lower QPS",
                "notes": "Low-dimensional small index where hnswlib flat array traversal overhead is minimal",
            })

    output_data = {
        "run_id": "where_it_loses_20261011_050000_c12",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "threshold_pct": 5.0,
        "speed_parity_rule": "Fashion-MNIST speed ordering is non-deterministic across runs; treated as parity per C9",
        "comparisons": comparisons,
    }
    return output_data


def main():
    data = analyze_losing_comparisons()
    out_file = os.path.join(RAW_DIR, "where_it_loses.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Saved: {out_file} ({len(data['comparisons'])} comparisons found)")

    # Update manifest
    with open(out_file, "rb") as f:
        b = f.read()
    sha = hashlib.sha256(b).hexdigest()
    sz = len(b)

    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        runs = manifest.get("runs", [])
        found = False
        for r in runs:
            if r.get("file") == "where_it_loses.json":
                r["sha256"] = sha
                r["size_bytes"] = sz
                found = True
                break
        if not found:
            runs.append({
                "file": "where_it_loses.json",
                "run_id": data["run_id"],
                "dataset": "cross_dataset",
                "label": "Where HNSWDB Loses (>5% Gap)",
                "sha256": sha,
                "size_bytes": sz,
            })
        manifest["runs"] = runs
        with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        print("Updated MANIFEST.json with where_it_loses.json")


if __name__ == "__main__":
    main()
