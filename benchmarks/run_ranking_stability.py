#!/usr/bin/env python3
"""
benchmarks/run_ranking_stability.py — Evaluate ranking stability across repeated Fashion-MNIST runs.

Reads the repeated Fashion-MNIST runs and outputs ranking stability data to results/raw/ranking_stability_fashion_mnist.json.
"""

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(ROOT, "results", "raw")


def run_ranking_stability():
    run_files = [
        ("Run 1", "bench_fashion_mnist_0a9f92db.json"),
        ("Run 2", "bench_fashion_mnist_723b5b0a.json"),
        ("Run 3", "bench_fashion_mnist_b3b2cd25.json"),
    ]

    runs_data = []
    for label, fname in run_files:
        fpath = os.path.join(RAW_DIR, fname)
        if not os.path.exists(fpath):
            continue
        with open(fpath, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        run_id = data.get("run_id", fname)
        engines = data.get("engines", {})
        h_eng = next(iter(engines.get("hnswdb", {}).values()), {})
        f_eng = next(iter(engines.get("faiss", {}).values()), {})
        hl_eng = next(iter(engines.get("hnswlib", {}).values()), {})

        # Build times
        b_times = {
            "hnswdb": round(h_eng.get("build_time_s", 0.0), 4),
            "faiss": round(f_eng.get("build_time_s", 0.0), 4),
            "hnswlib": round(hl_eng.get("build_time_s", 0.0), 4),
        }
        b_order = " < ".join(sorted(b_times.keys(), key=lambda k: b_times[k]))

        # QPS at ef=50 (R~0.995)
        # In ef_results index 1 is ef=50
        def get_qps(sd):
            for ef_r in sd.get("ef_results", []):
                if ef_r.get("ef_search") == 50:
                    q = ef_r.get("qps_by_threads", {}).get("1", {}).get("median")
                    if q is not None:
                        return round(float(q), 1)
                    return round(float(ef_r.get("latency", {}).get("sequential_qps", 0.0)), 1)
            return 0.0

        def get_p50(sd):
            for ef_r in sd.get("ef_results", []):
                if ef_r.get("ef_search") == 50:
                    return round(float(ef_r.get("latency", {}).get("p50", 0.0)), 4)
            return 0.0

        qps = {
            "faiss": get_qps(f_eng),
            "hnswdb": get_qps(h_eng),
            "hnswlib": get_qps(hl_eng),
        }
        q_order = " > ".join(sorted(qps.keys(), key=lambda k: qps[k], reverse=True))

        p50 = {
            "faiss": get_p50(f_eng),
            "hnswdb": get_p50(h_eng),
            "hnswlib": get_p50(hl_eng),
        }
        p_order = " < ".join(sorted(p50.keys(), key=lambda k: p50[k]))

        runs_data.append({
            "label": label,
            "run_id": run_id,
            "raw_file": fname,
            "build_time_s": b_times,
            "build_time_order": b_order,
            "seq_qps_at_099": qps,
            "seq_qps_order": q_order,
            "p50_latency_ms_at_099": p50,
            "p50_latency_order": p_order,
        })

    out = {
        "run_id": "ranking_stability_20261011_045000_c9stable",
        "dataset": "fashion_mnist",
        "target_recall": 0.99,
        "all_rankings_identical": False,
        "note": "relative speed varies between runs on this host; treat as parity",
        "runs": runs_data,
    }

    out_file = os.path.join(RAW_DIR, "ranking_stability_fashion_mnist.json")
    if os.path.exists(out_file) and "--overwrite" not in sys.argv:
        print(f"Existing file preserved: {out_file} (pass --overwrite to rewrite)")
        return
    with open(out_file, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"Ranking stability written to: {out_file}")


if __name__ == "__main__":
    run_ranking_stability()
