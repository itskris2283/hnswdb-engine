"""
Phase 3D: Failure Mechanism Decomposition.
Isolates and quantifies the step-by-step failure mechanism on clustered data:
1. Entry point descent and beam search candidate quality during construction
2. Reverse-link retention and severance during overflow reselection
3. Evolution of in-degrees, zero-in-degree nodes, and unreachable components as nodes are inserted
Compares legacy HNSWDB configuration vs cured configuration.
Saves results to results/raw/decomposition_dataset_b.json.
"""

import os
import sys
import json
import time
import uuid
import numpy as np

import hnswdb
from hnswdb import HNSWIndex, BuildOptions
from benchmarks.datasets_synth import get_dataset_b, compute_ground_truth_bruteforce
from benchmarks.metrics import recall_at_k


def run_decomposition():
    run_id = f"decomp-{uuid.uuid4().hex[:8]}"
    print(f"Starting Phase 3D Failure Mechanism Decomposition [run_id: {run_id}]...")

    xb, xq, gt_nbrs, _, _ = get_dataset_b(seed=42)
    n = len(xb)
    dim = xb.shape[1]
    ids = np.arange(n, dtype=np.int64)

    opt_legacy = BuildOptions("M0", True, False, "heuristic", "closest")
    opt_cured = BuildOptions("M", False, False, "heuristic", "closest")

    configs = [
        ("legacy", opt_legacy),
        ("cured", opt_cured),
    ]

    results = {
        "run_id": run_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset": "Dataset B (20K x 64d, 50 clusters, sigma=0.1)",
        "comparisons": {}
    }

    # Step-wise milestones: 1K, 2K, 5K, 10K, 20K
    milestones = [1000, 2000, 5000, 10000, 20000]

    for name, opt in configs:
        print(f"\nAnalyzing configuration: {name}...")
        idx = HNSWIndex(dim=dim, metric="l2", max_elements=n, M=16, ef_construction=100, seed=42, build_options=opt)

        trajectory = []
        cur_idx = 0

        for ms in milestones:
            chunk = xb[cur_idx:ms]
            chunk_ids = ids[cur_idx:ms]
            idx.add(chunk, chunk_ids, num_threads=1)
            cur_idx = ms

            diag = idx.diagnostics()
            unreach = idx.get_unreachable_count()
            zero_in = diag.get("zero_in_degree_count", 0)
            dups = diag.get("duplicate_links_count", 0)
            mean_deg = diag.get("layer0_degree_mean", 0.0)
            wcc = diag.get("weakly_connected_components", 0)
            recip = diag.get("reciprocal_links_fraction", 0.0)

            print(f"  Milestone {ms:5d}: unreachable={unreach:4d}, zero_in={int(zero_in):4d}, mean_deg={mean_deg:.1f}, recip={recip:.3f}, wcc={int(wcc)}")
            trajectory.append({
                "elements": ms,
                "unreachable_nodes": int(unreach),
                "zero_in_degree_nodes": int(zero_in),
                "duplicate_links": int(dups),
                "mean_layer0_degree": float(mean_deg),
                "reciprocal_links_fraction": float(recip),
                "weakly_connected_components": int(wcc),
            })

        # Final query recall evaluation
        pred_ids_10, _ = idx.search(xq, k=10, ef_search=10, num_threads=1)
        pred_ids_50, _ = idx.search(xq, k=10, ef_search=50, num_threads=1)
        pred_ids_200, _ = idx.search(xq, k=10, ef_search=200, num_threads=1)

        rec_10 = recall_at_k(pred_ids_10, gt_nbrs, 10)
        rec_50 = recall_at_k(pred_ids_50, gt_nbrs, 10)
        rec_200 = recall_at_k(pred_ids_200, gt_nbrs, 10)

        # Inspect isolated nodes: analyze why unreachable nodes are missed
        in_degrees = idx.get_in_degrees_l0()
        in_deg_arr = np.array(in_degrees[:n])
        in_deg_hist = {
            "deg_0": int(np.sum(in_deg_arr == 0)),
            "deg_1_to_5": int(np.sum((in_deg_arr >= 1) & (in_deg_arr <= 5))),
            "deg_6_to_15": int(np.sum((in_deg_arr >= 6) & (in_deg_arr <= 15))),
            "deg_16_to_32": int(np.sum((in_deg_arr >= 16) & (in_deg_arr <= 32))),
            "deg_gt_32": int(np.sum(in_deg_arr > 32)),
        }

        results["comparisons"][name] = {
            "options": {
                "new_node_degree": opt.new_node_degree,
                "keep_pruned": opt.keep_pruned,
                "protect_last_in_edge": opt.protect_last_in_edge,
            },
            "recalls": {
                "ef10": rec_10,
                "ef50": rec_50,
                "ef200": rec_200,
            },
            "final_unreachable": trajectory[-1]["unreachable_nodes"],
            "final_zero_in_degree": trajectory[-1]["zero_in_degree_nodes"],
            "trajectory": trajectory,
            "in_degree_distribution": in_deg_hist,
        }

    out_path = os.path.join("results", "raw", "decomposition_dataset_b.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n[DONE] Saved failure mechanism decomposition to {out_path}")
    return results


if __name__ == "__main__":
    run_decomposition()
