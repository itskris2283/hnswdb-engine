"""
Phase 3B: Ablation Experiments on Dataset B.
Evaluates full factorial design over C++ BuildOptions:
- new_node_degree: ["M0", "M"]
- keep_pruned: [True, False]
- protect_last_in_edge: [False, True]
- overflow_selection: ["heuristic", "closest"]
- layer_entry: ["closest", "all"]

Records recall@10 at ef in [10, 20, 50, 100, 200], unreachable nodes, build time on Dataset B,
and cross-validates on Fashion-MNIST (5K train, 1K test) to verify guardrails.
Saves results to results/raw/ablations_dataset_b.json.
"""

import os
import sys
import json
import time
import uuid
import itertools
import numpy as np

import hnswdb
from hnswdb import HNSWIndex, BuildOptions
from benchmarks.datasets_synth import get_dataset_b, get_dataset_c, compute_ground_truth_bruteforce
from benchmarks.metrics import recall_at_k


def run_ablations():
    run_id = f"ablations-{uuid.uuid4().hex[:8]}"
    print(f"Starting Phase 3B Ablations [run_id: {run_id}]...")

    print("Loading Dataset B...")
    xb_b, xq_b, gt_nbrs_b, _, _ = get_dataset_b(seed=42)
    n_b, dim_b = xb_b.shape
    ids_b = np.arange(n_b, dtype=np.int64)

    print("Loading Fashion-MNIST subset (5K vectors)...")
    xb_c_full, xq_c_full, _, _, _ = get_dataset_c()
    xb_fmn = xb_c_full[:5000]
    xq_fmn = xq_c_full[:500]
    gt_nbrs_fmn, _ = compute_ground_truth_bruteforce(xb_fmn, xq_fmn, metric="l2", k=10)
    ids_fmn = np.arange(len(xb_fmn), dtype=np.int64)

    # Core 8 factorial combinations
    core_factors = [
        ("new_node_degree", ["M0", "M"]),
        ("keep_pruned", [True, False]),
        ("protect_last_in_edge", [False, True]),
    ]

    configs = []
    # 2^3 = 8 core configs (with overflow_selection="heuristic", layer_entry="closest")
    for nnd, kp, plie in itertools.product(["M0", "M"], [True, False], [False, True]):
        configs.append({
            "name": f"degree_{nnd}__keep_{kp}__protect_{plie}",
            "new_node_degree": nnd,
            "keep_pruned": kp,
            "protect_last_in_edge": plie,
            "overflow_selection": "heuristic",
            "layer_entry": "closest",
        })

    # Additional exploratory variants
    configs.append({
        "name": "degree_M__keep_False__overflow_closest",
        "new_node_degree": "M",
        "keep_pruned": False,
        "protect_last_in_edge": False,
        "overflow_selection": "closest",
        "layer_entry": "closest",
    })
    configs.append({
        "name": "degree_M__keep_False__entry_all",
        "new_node_degree": "M",
        "keep_pruned": False,
        "protect_last_in_edge": False,
        "overflow_selection": "heuristic",
        "layer_entry": "all",
    })

    results = {
        "run_id": run_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset_b": "20K x 64d, 50 clusters, sigma=0.1",
        "dataset_fmn": "Fashion-MNIST 5K subset x 784d",
        "ef_values": [10, 20, 50, 100, 200],
        "configs": []
    }

    ef_list = [10, 20, 50, 100, 200]

    for cfg in configs:
        name = cfg["name"]
        print(f"\nEvaluating config: {name}...")
        opt = BuildOptions(
            new_node_degree=cfg["new_node_degree"],
            keep_pruned=cfg["keep_pruned"],
            protect_last_in_edge=cfg["protect_last_in_edge"],
            overflow_selection=cfg["overflow_selection"],
            layer_entry=cfg["layer_entry"]
        )

        # 1. Evaluate on Dataset B
        idx_b = HNSWIndex(dim=dim_b, metric="l2", max_elements=n_b, M=16, ef_construction=100, seed=42, build_options=opt)
        t0 = time.perf_counter()
        idx_b.add(xb_b, ids_b, num_threads=1)
        b_build_time = time.perf_counter() - t0

        b_recalls = {}
        for ef in ef_list:
            pred_ids, _ = idx_b.search(xq_b, k=10, ef_search=ef, num_threads=1)
            b_recalls[f"recall_ef{ef}"] = recall_at_k(pred_ids, gt_nbrs_b, 10)

        unreach_b = idx_b.get_unreachable_count()
        diag_b = idx_b.diagnostics()

        # 2. Evaluate on Fashion-MNIST subset
        idx_fmn = HNSWIndex(dim=784, metric="l2", max_elements=len(xb_fmn), M=16, ef_construction=100, seed=42, build_options=opt)
        t0 = time.perf_counter()
        idx_fmn.add(xb_fmn, ids_fmn, num_threads=1)
        fmn_build_time = time.perf_counter() - t0

        pred_fmn_50, _ = idx_fmn.search(xq_fmn, k=10, ef_search=50, num_threads=1)
        pred_fmn_100, _ = idx_fmn.search(xq_fmn, k=10, ef_search=100, num_threads=1)
        fmn_rec_50 = recall_at_k(pred_fmn_50, gt_nbrs_fmn, 10)
        fmn_rec_100 = recall_at_k(pred_fmn_100, gt_nbrs_fmn, 10)

        entry = {
            "name": name,
            "options": cfg,
            "dataset_b": {
                "build_time_s": b_build_time,
                "recalls": b_recalls,
                "unreachable_nodes": int(unreach_b),
                "zero_in_degree": int(diag_b.get("zero_in_degree_count", 0)),
                "duplicate_links": int(diag_b.get("duplicate_links_count", 0)),
                "mean_degree": float(diag_b.get("layer0_degree_mean", 0.0)),
            },
            "fashion_mnist": {
                "build_time_s": fmn_build_time,
                "recall_ef50": fmn_rec_50,
                "recall_ef100": fmn_rec_100,
            }
        }

        # Check targets T2-T5
        t2_pass = b_recalls["recall_ef10"] >= 0.880
        t3_pass = b_recalls["recall_ef20"] >= 0.960
        t4_pass = b_recalls["recall_ef50"] >= 0.985
        t5_pass = b_recalls["recall_ef200"] >= 0.995

        entry["targets_check"] = {
            "T2_ef10_ge_0.880": {"target": 0.880, "val": b_recalls["recall_ef10"], "pass": bool(t2_pass)},
            "T3_ef20_ge_0.960": {"target": 0.960, "val": b_recalls["recall_ef20"], "pass": bool(t3_pass)},
            "T4_ef50_ge_0.985": {"target": 0.985, "val": b_recalls["recall_ef50"], "pass": bool(t4_pass)},
            "T5_ef200_ge_0.995": {"target": 0.995, "val": b_recalls["recall_ef200"], "pass": bool(t5_pass)},
            "all_passed": bool(t2_pass and t3_pass and t4_pass and t5_pass)
        }

        print(f"  B recalls: ef10={b_recalls['recall_ef10']:.4f}, ef50={b_recalls['recall_ef50']:.4f}, ef200={b_recalls['recall_ef200']:.4f} | Unreachable={unreach_b} | Build={b_build_time:.2f}s")
        print(f"  F-MNIST: ef50={fmn_rec_50:.4f}, ef100={fmn_rec_100:.4f} | Build={fmn_build_time:.2f}s")
        results["configs"].append(entry)

    out_path = os.path.join("results", "raw", "ablations_dataset_b.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n[DONE] Saved ablation results to {out_path}")
    return results


if __name__ == "__main__":
    run_ablations()
