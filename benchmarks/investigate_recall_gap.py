#!/usr/bin/env python3
"""
Phase D: Root-cause investigation for clustered-data recall gap.

HNSWDB gets ~0.984 recall@10 at ef=200 on Dataset B (20K x 64d, 50 clusters, sigma=0.1)
while FAISS gets ~0.999 and hnswlib gets ~1.0 at identical M=16, efC=100.

This script tests specific hypotheses:
1. Distance function parity: verify HNSWDB/FAISS/hnswlib compute the same distances
2. Graph connectivity: count reachable nodes from entry point
3. Neighbor quality: average distance to neighbors vs optimal
4. efC sensitivity: recall vs efC curve (does HNSWDB eventually converge?)
5. Search exhaustiveness: track how many nodes are visited during search
"""

import os
import sys
import json
import time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import hnswdb
from benchmarks.datasets_synth import get_dataset_b, compute_ground_truth_bruteforce
from benchmarks.metrics import recall_at_k


def get_dataset_b_sigma(sigma=0.1, seed=42):
    """Dataset B variant with configurable sigma."""
    rng = np.random.default_rng(seed)
    n_train, n_test, dim, n_clusters = 20000, 1000, 64, 50
    centroids = rng.standard_normal((n_clusters, dim)).astype(np.float32)
    train_labels = rng.integers(0, n_clusters, size=n_train)
    noise_train = rng.standard_normal((n_train, dim)).astype(np.float32) * sigma
    xb = (centroids[train_labels] + noise_train).astype(np.float32)
    test_labels = rng.integers(0, n_clusters, size=n_test)
    noise_test = rng.standard_normal((n_test, dim)).astype(np.float32) * sigma
    xq = (centroids[test_labels] + noise_test).astype(np.float32)
    gt_nbrs, gt_dists = compute_ground_truth_bruteforce(xb, xq, metric="l2", k=100)
    return xb, xq, gt_nbrs, gt_dists


def hypothesis_1_distance_parity(xb, xq):
    """Test: Does HNSWDB compute the same distances as brute force?"""
    print("\n=== Hypothesis 1: Distance function parity ===")
    idx = hnswdb.HNSWIndex(dim=xb.shape[1], metric="l2", max_elements=len(xb), M=16, ef_construction=100, seed=42)
    ids = np.arange(len(xb), dtype=np.int64)
    idx.add(xb, ids, num_threads=1)

    # Search a few queries and compare returned distances
    n_check = 50
    max_err = 0.0
    for i in range(n_check):
        q = np.ascontiguousarray(xq[i:i+1], dtype=np.float32)
        res_ids, res_dists = idx.search(q, k=10, ef_search=200)
        for j in range(min(10, len(res_ids[0]))):
            rid = res_ids[0][j]
            if rid < 0:
                continue
            # Compute expected distance
            diff = xq[i] - xb[rid]
            expected = float(np.sum(diff * diff))
            actual = float(res_dists[0][j])
            rel_err = abs(actual - expected) / max(expected, 1e-12)
            if rel_err > max_err:
                max_err = rel_err
    
    print(f"  Max relative distance error over {n_check} queries: {max_err:.2e}")
    print(f"  VERDICT: {'PASS' if max_err < 1e-4 else 'FAIL'} (threshold: 1e-4)")
    return {"max_rel_error": max_err, "pass": max_err < 1e-4}


def hypothesis_2_efc_sensitivity(xb, xq, gt_nbrs):
    """Test: How does recall change with efC? Does HNSWDB converge to 1.0?"""
    print("\n=== Hypothesis 2: efC sensitivity ===")
    efc_values = [50, 100, 200, 400, 800]
    results = {}
    
    for efc in efc_values:
        idx = hnswdb.HNSWIndex(dim=xb.shape[1], metric="l2", max_elements=len(xb), M=16, ef_construction=efc, seed=42)
        ids = np.arange(len(xb), dtype=np.int64)
        idx.add(xb, ids, num_threads=1)
        
        q = np.ascontiguousarray(xq, dtype=np.float32)
        res_ids, _ = idx.search(q, k=10, ef_search=200)
        recall = recall_at_k(res_ids, gt_nbrs, 10)
        results[efc] = recall
        print(f"  efC={efc:>4}: recall@10 (ef=200) = {recall:.4f}")
    
    converging = results[800] > results[100] + 0.005
    print(f"  VERDICT: {'Converging' if converging else 'Flat'} — efC=800 recall {results[800]:.4f} vs efC=100 recall {results[100]:.4f}")
    return results


def hypothesis_3_seed_sensitivity(xb, xq, gt_nbrs):
    """Test: How much does recall vary across seeds?"""
    print("\n=== Hypothesis 3: Seed sensitivity ===")
    seeds = [0, 1, 2, 3, 42, 100, 999]
    recalls = []
    
    for seed in seeds:
        idx = hnswdb.HNSWIndex(dim=xb.shape[1], metric="l2", max_elements=len(xb), M=16, ef_construction=100, seed=seed)
        ids = np.arange(len(xb), dtype=np.int64)
        idx.add(xb, ids, num_threads=1)
        
        q = np.ascontiguousarray(xq, dtype=np.float32)
        res_ids, _ = idx.search(q, k=10, ef_search=200)
        recall = recall_at_k(res_ids, gt_nbrs, 10)
        recalls.append(recall)
        print(f"  seed={seed:>4}: recall@10 (ef=200) = {recall:.4f}")
    
    print(f"  Range: {min(recalls):.4f} - {max(recalls):.4f} (std={np.std(recalls):.4f})")
    return {"seeds": seeds, "recalls": recalls, "std": float(np.std(recalls))}


def hypothesis_4_sigma_sweep(xq_fn, gt_fn):
    """Test: How does recall scale with cluster tightness?"""
    print("\n=== Hypothesis 4: Sigma sweep ===")
    sigmas = [0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0]
    results = {}
    
    for sigma in sigmas:
        xb, xq, gt_nbrs, _ = get_dataset_b_sigma(sigma=sigma, seed=42)
        idx = hnswdb.HNSWIndex(dim=xb.shape[1], metric="l2", max_elements=len(xb), M=16, ef_construction=100, seed=42)
        ids = np.arange(len(xb), dtype=np.int64)
        idx.add(xb, ids, num_threads=1)
        
        q = np.ascontiguousarray(xq, dtype=np.float32)
        res_ids, _ = idx.search(q, k=10, ef_search=200)
        recall = recall_at_k(res_ids, gt_nbrs, 10)
        results[sigma] = recall
        print(f"  sigma={sigma:.2f}: recall@10 (ef=200) = {recall:.4f}")
    
    return results


def hypothesis_5_graph_diagnostics(xb):
    """Test: Graph quality metrics from diagnostics()."""
    print("\n=== Hypothesis 5: Graph diagnostics ===")
    idx = hnswdb.HNSWIndex(dim=xb.shape[1], metric="l2", max_elements=len(xb), M=16, ef_construction=100, seed=42)
    ids = np.arange(len(xb), dtype=np.int64)
    idx.add(xb, ids, num_threads=1)
    
    diag = idx.diagnostics()
    for k, v in sorted(diag.items()):
        print(f"  {k}: {v}")
    return diag


def hypothesis_6_missed_queries(xb, xq, gt_nbrs):
    """Test: Which queries have low recall and what characterizes them?"""
    print("\n=== Hypothesis 6: Per-query recall analysis ===")
    idx = hnswdb.HNSWIndex(dim=xb.shape[1], metric="l2", max_elements=len(xb), M=16, ef_construction=100, seed=42)
    ids = np.arange(len(xb), dtype=np.int64)
    idx.add(xb, ids, num_threads=1)
    
    q = np.ascontiguousarray(xq, dtype=np.float32)
    res_ids, _ = idx.search(q, k=10, ef_search=200)
    
    # Per-query recall
    per_q_recall = []
    for i in range(len(xq)):
        found = set(res_ids[i][:10].tolist())
        truth = set(gt_nbrs[i][:10].tolist())
        found.discard(-1)
        r = len(found & truth) / max(len(truth), 1)
        per_q_recall.append(r)
    
    per_q_recall = np.array(per_q_recall)
    n_perfect = np.sum(per_q_recall >= 1.0 - 1e-6)
    n_below_90 = np.sum(per_q_recall < 0.9)
    n_below_50 = np.sum(per_q_recall < 0.5)
    
    print(f"  Perfect recall (10/10): {n_perfect}/{len(xq)} ({100*n_perfect/len(xq):.1f}%)")
    print(f"  Recall < 0.9: {n_below_90}/{len(xq)}")
    print(f"  Recall < 0.5: {n_below_50}/{len(xq)}")
    print(f"  Mean: {per_q_recall.mean():.4f}, Std: {per_q_recall.std():.4f}")
    print(f"  Min: {per_q_recall.min():.4f} at query #{np.argmin(per_q_recall)}")
    
    return {
        "n_perfect": int(n_perfect),
        "n_below_90": int(n_below_90),
        "n_below_50": int(n_below_50),
        "mean": float(per_q_recall.mean()),
        "min": float(per_q_recall.min()),
    }


def hypothesis_7_M_sensitivity(xb, xq, gt_nbrs):
    """Test: Does increasing M fix the recall gap?"""
    print("\n=== Hypothesis 7: M sensitivity ===")
    M_values = [8, 16, 24, 32, 48]
    results = {}
    
    for M in M_values:
        idx = hnswdb.HNSWIndex(dim=xb.shape[1], metric="l2", max_elements=len(xb), M=M, ef_construction=100, seed=42)
        ids = np.arange(len(xb), dtype=np.int64)
        idx.add(xb, ids, num_threads=1)
        
        q = np.ascontiguousarray(xq, dtype=np.float32)
        res_ids, _ = idx.search(q, k=10, ef_search=200)
        recall = recall_at_k(res_ids, gt_nbrs, 10)
        results[M] = recall
        print(f"  M={M:>2}: recall@10 (ef=200, efC=100) = {recall:.4f}")
    
    return results


def main():
    print("=" * 60)
    print("Phase D: Root-Cause Investigation — Clustered Data Recall Gap")
    print("=" * 60)
    
    # Load Dataset B (sigma=0.1)
    print("\nLoading Dataset B (sigma=0.1, seed=42)...")
    xb, xq, gt_nbrs, gt_dists = get_dataset_b_sigma(sigma=0.1, seed=42)
    print(f"  xb: {xb.shape}, xq: {xq.shape}")
    
    results = {}
    
    results["h1_distance_parity"] = hypothesis_1_distance_parity(xb, xq)
    results["h2_efc_sensitivity"] = hypothesis_2_efc_sensitivity(xb, xq, gt_nbrs)
    results["h3_seed_sensitivity"] = hypothesis_3_seed_sensitivity(xb, xq, gt_nbrs)
    results["h4_sigma_sweep"] = hypothesis_4_sigma_sweep(None, None)
    results["h5_diagnostics"] = hypothesis_5_graph_diagnostics(xb)
    results["h6_per_query"] = hypothesis_6_missed_queries(xb, xq, gt_nbrs)
    results["h7_M_sensitivity"] = hypothesis_7_M_sensitivity(xb, xq, gt_nbrs)
    
    # Save results
    raw_dir = os.path.join(ROOT, "results", "raw")
    os.makedirs(raw_dir, exist_ok=True)
    out_path = os.path.join(raw_dir, "investigation_recall_gap.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
