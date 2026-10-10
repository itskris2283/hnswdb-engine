"""
Phase 3A: 5x5 Seed Matrix Experiment on Dataset B.
Separates data_seed in {0..4} and build_seed in {0..4} (25 combinations)
across HNSWDB (legacy baseline & cured), hnswlib, and FAISS.
Measures recall@10 at ef=200, unreachable node counts, single-thread vs 4-thread builds,
and computes two-way variance decomposition (ANOVA).
"""

import os
import sys
import json
import time
import uuid
import numpy as np

import hnswdb
from hnswdb import HNSWIndex, BuildOptions
import hnswlib
import faiss
from benchmarks.datasets_synth import compute_ground_truth_bruteforce
from benchmarks.metrics import recall_at_k


def generate_dataset_b(data_seed: int):
    rng = np.random.default_rng(data_seed)
    n_train, n_test, dim, n_clusters = 20000, 1000, 64, 50
    centroids = rng.standard_normal((n_clusters, dim)).astype(np.float32)
    train_labels = rng.integers(0, n_clusters, size=n_train)
    noise_train = rng.standard_normal((n_train, dim)).astype(np.float32) * 0.1
    xb = (centroids[train_labels] + noise_train).astype(np.float32)

    test_labels = rng.integers(0, n_clusters, size=n_test)
    noise_test = rng.standard_normal((n_test, dim)).astype(np.float32) * 0.1
    xq = (centroids[test_labels] + noise_test).astype(np.float32)

    gt_nbrs, _ = compute_ground_truth_bruteforce(xb, xq, metric="l2", k=10)
    return xb, xq, gt_nbrs


def compute_two_way_anova(matrix: np.ndarray):
    """
    matrix: shape (I, J) where I is data_seed, J is build_seed
    Returns dict with SS, df, MS, F, and variance shares
    """
    I, J = matrix.shape
    grand_mean = np.mean(matrix)
    row_means = np.mean(matrix, axis=1)
    col_means = np.mean(matrix, axis=0)

    ss_total = np.sum((matrix - grand_mean) ** 2)
    ss_data = J * np.sum((row_means - grand_mean) ** 2)
    ss_build = I * np.sum((col_means - grand_mean) ** 2)
    ss_error = max(0.0, ss_total - ss_data - ss_build)

    eps = 1e-12
    return {
        "grand_mean": float(grand_mean),
        "ss_total": float(ss_total),
        "ss_data": float(ss_data),
        "ss_build": float(ss_build),
        "ss_error": float(ss_error),
        "share_data_percent": float((ss_data / (ss_total + eps)) * 100.0),
        "share_build_percent": float((ss_build / (ss_total + eps)) * 100.0),
        "share_error_percent": float((ss_error / (ss_total + eps)) * 100.0),
    }


def run_seed_matrix():
    data_seeds = [0, 1, 2, 3, 4]
    build_seeds = [0, 1, 2, 3, 4]

    run_id = f"seed-matrix-{uuid.uuid4().hex[:8]}"
    print(f"Starting Phase 3A 5x5 Seed Matrix [run_id: {run_id}]...")

    results = {
        "run_id": run_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset": "Dataset B (20K x 64d, 50 clusters, sigma=0.1)",
        "params": {"M": 16, "ef_construction": 100, "ef_search": 200, "k": 10},
        "data_seeds": data_seeds,
        "build_seeds": build_seeds,
        "runs": [],
        "engines": {}
    }

    engine_recalls_1t = {
        "hnswdb_legacy": np.zeros((len(data_seeds), len(build_seeds)), dtype=np.float64),
        "hnswdb_cured": np.zeros((len(data_seeds), len(build_seeds)), dtype=np.float64),
        "hnswlib": np.zeros((len(data_seeds), len(build_seeds)), dtype=np.float64),
        "faiss": np.zeros((len(data_seeds), len(build_seeds)), dtype=np.float64),
    }

    engine_recalls_4t = {
        "hnswdb_legacy": np.zeros((len(data_seeds), len(build_seeds)), dtype=np.float64),
        "hnswdb_cured": np.zeros((len(data_seeds), len(build_seeds)), dtype=np.float64),
        "hnswlib": np.zeros((len(data_seeds), len(build_seeds)), dtype=np.float64),
        "faiss": np.zeros((len(data_seeds), len(build_seeds)), dtype=np.float64),
    }

    unreachable_matrix_legacy = np.zeros((len(data_seeds), len(build_seeds)), dtype=np.int64)
    unreachable_matrix_cured = np.zeros((len(data_seeds), len(build_seeds)), dtype=np.int64)

    opt_legacy = BuildOptions("M0", True, False, "heuristic", "closest")
    opt_cured = BuildOptions("M", False, False, "heuristic", "closest")

    for i, d_seed in enumerate(data_seeds):
        print(f"\nGenerating Dataset B with data_seed={d_seed}...")
        xb, xq, gt_nbrs = generate_dataset_b(d_seed)
        n, dim = xb.shape
        ids_arr = np.arange(n, dtype=np.int64)

        for j, b_seed in enumerate(build_seeds):
            print(f"  Running build_seed={b_seed}...", end="", flush=True)

            run_entry = {
                "data_seed": d_seed,
                "build_seed": b_seed,
                "hnswdb_legacy": {},
                "hnswdb_cured": {},
                "hnswlib": {},
                "faiss": {},
            }

            # 1. HNSWDB Legacy (1-thread)
            idx_leg_1t = HNSWIndex(dim=dim, metric="l2", max_elements=n, M=16, ef_construction=100, seed=b_seed, build_options=opt_legacy)
            t0 = time.perf_counter()
            idx_leg_1t.add(xb, ids_arr, num_threads=1)
            t_leg_1t = time.perf_counter() - t0
            pred_ids, _ = idx_leg_1t.search(xq, k=10, ef_search=200, num_threads=1)
            rec_leg_1t = recall_at_k(pred_ids, gt_nbrs, 10)
            unreach_leg = idx_leg_1t.get_unreachable_count()

            # HNSWDB Legacy (4-thread)
            idx_leg_4t = HNSWIndex(dim=dim, metric="l2", max_elements=n, M=16, ef_construction=100, seed=b_seed, build_options=opt_legacy)
            t0 = time.perf_counter()
            idx_leg_4t.add(xb, ids_arr, num_threads=4)
            t_leg_4t = time.perf_counter() - t0
            pred_ids_4t, _ = idx_leg_4t.search(xq, k=10, ef_search=200, num_threads=4)
            rec_leg_4t = recall_at_k(pred_ids_4t, gt_nbrs, 10)

            run_entry["hnswdb_legacy"] = {
                "recall_1t": rec_leg_1t,
                "recall_4t": rec_leg_4t,
                "build_time_1t_s": t_leg_1t,
                "build_time_4t_s": t_leg_4t,
                "unreachable_nodes": int(unreach_leg),
            }
            engine_recalls_1t["hnswdb_legacy"][i, j] = rec_leg_1t
            engine_recalls_4t["hnswdb_legacy"][i, j] = rec_leg_4t
            unreachable_matrix_legacy[i, j] = unreach_leg

            # 2. HNSWDB Cured (1-thread)
            idx_cur_1t = HNSWIndex(dim=dim, metric="l2", max_elements=n, M=16, ef_construction=100, seed=b_seed, build_options=opt_cured)
            t0 = time.perf_counter()
            idx_cur_1t.add(xb, ids_arr, num_threads=1)
            t_cur_1t = time.perf_counter() - t0
            pred_ids_c, _ = idx_cur_1t.search(xq, k=10, ef_search=200, num_threads=1)
            rec_cur_1t = recall_at_k(pred_ids_c, gt_nbrs, 10)
            unreach_cur = idx_cur_1t.get_unreachable_count()

            # HNSWDB Cured (4-thread)
            idx_cur_4t = HNSWIndex(dim=dim, metric="l2", max_elements=n, M=16, ef_construction=100, seed=b_seed, build_options=opt_cured)
            t0 = time.perf_counter()
            idx_cur_4t.add(xb, ids_arr, num_threads=4)
            t_cur_4t = time.perf_counter() - t0
            pred_ids_c4, _ = idx_cur_4t.search(xq, k=10, ef_search=200, num_threads=4)
            rec_cur_4t = recall_at_k(pred_ids_c4, gt_nbrs, 10)

            run_entry["hnswdb_cured"] = {
                "recall_1t": rec_cur_1t,
                "recall_4t": rec_cur_4t,
                "build_time_1t_s": t_cur_1t,
                "build_time_4t_s": t_cur_4t,
                "unreachable_nodes": int(unreach_cur),
            }
            engine_recalls_1t["hnswdb_cured"][i, j] = rec_cur_1t
            engine_recalls_4t["hnswdb_cured"][i, j] = rec_cur_4t
            unreachable_matrix_cured[i, j] = unreach_cur

            # 3. hnswlib (1-thread)
            hl_1t = hnswlib.Index(space="l2", dim=dim)
            hl_1t.init_index(max_elements=n, ef_construction=100, M=16, random_seed=b_seed)
            t0 = time.perf_counter()
            hl_1t.add_items(xb, ids_arr, num_threads=1)
            t_hl_1t = time.perf_counter() - t0
            hl_1t.set_ef(200)
            hl_ids_1t, _ = hl_1t.knn_query(xq, k=10, num_threads=1)
            rec_hl_1t = recall_at_k(hl_ids_1t, gt_nbrs, 10)

            # hnswlib (4-thread)
            hl_4t = hnswlib.Index(space="l2", dim=dim)
            hl_4t.init_index(max_elements=n, ef_construction=100, M=16, random_seed=b_seed)
            t0 = time.perf_counter()
            hl_4t.add_items(xb, ids_arr, num_threads=4)
            t_hl_4t = time.perf_counter() - t0
            hl_4t.set_ef(200)
            hl_ids_4t, _ = hl_4t.knn_query(xq, k=10, num_threads=4)
            rec_hl_4t = recall_at_k(hl_ids_4t, gt_nbrs, 10)

            run_entry["hnswlib"] = {
                "recall_1t": rec_hl_1t,
                "recall_4t": rec_hl_4t,
                "build_time_1t_s": t_hl_1t,
                "build_time_4t_s": t_hl_4t,
            }
            engine_recalls_1t["hnswlib"][i, j] = rec_hl_1t
            engine_recalls_4t["hnswlib"][i, j] = rec_hl_4t

            # 4. FAISS (1-thread)
            faiss.omp_set_num_threads(1)
            fi_1t = faiss.IndexHNSWFlat(dim, 16, faiss.METRIC_L2)
            fi_1t.hnsw.efConstruction = 100
            fi_1t.hnsw.rng = faiss.RandomGenerator(b_seed)
            t0 = time.perf_counter()
            fi_1t.add(xb)
            t_fi_1t = time.perf_counter() - t0
            fi_1t.hnsw.efSearch = 200
            _, fi_ids_1t = fi_1t.search(xq, 10)
            rec_fi_1t = recall_at_k(fi_ids_1t, gt_nbrs, 10)

            # FAISS (4-thread)
            faiss.omp_set_num_threads(4)
            fi_4t = faiss.IndexHNSWFlat(dim, 16, faiss.METRIC_L2)
            fi_4t.hnsw.efConstruction = 100
            fi_4t.hnsw.rng = faiss.RandomGenerator(b_seed)
            t0 = time.perf_counter()
            fi_4t.add(xb)
            t_fi_4t = time.perf_counter() - t0
            fi_4t.hnsw.efSearch = 200
            _, fi_ids_4t = fi_4t.search(xq, 10)
            rec_fi_4t = recall_at_k(fi_ids_4t, gt_nbrs, 10)

            run_entry["faiss"] = {
                "recall_1t": rec_fi_1t,
                "recall_4t": rec_fi_4t,
                "build_time_1t_s": t_fi_1t,
                "build_time_4t_s": t_fi_4t,
            }
            engine_recalls_1t["faiss"][i, j] = rec_fi_1t
            engine_recalls_4t["faiss"][i, j] = rec_fi_4t

            results["runs"].append(run_entry)
            print(f" done (leg: {rec_leg_1t:.4f}, cur: {rec_cur_1t:.4f}, hl: {rec_hl_1t:.4f}, fa: {rec_fi_1t:.4f})")

    # Aggregate summaries for each engine
    for eng_name, mat in engine_recalls_1t.items():
        mat_4t = engine_recalls_4t[eng_name]
        anova_res = compute_two_way_anova(mat)
        summary = {
            "recall_1t_mean": float(np.mean(mat)),
            "recall_1t_std": float(np.std(mat)),
            "recall_1t_min": float(np.min(mat)),
            "recall_1t_max": float(np.max(mat)),
            "recall_4t_mean": float(np.mean(mat_4t)),
            "recall_4t_std": float(np.std(mat_4t)),
            "recall_4t_min": float(np.min(mat_4t)),
            "recall_4t_max": float(np.max(mat_4t)),
            "anova_1t": anova_res,
        }
        if eng_name == "hnswdb_legacy":
            summary["unreachable_mean"] = float(np.mean(unreachable_matrix_legacy))
            summary["unreachable_min"] = int(np.min(unreachable_matrix_legacy))
            summary["unreachable_max"] = int(np.max(unreachable_matrix_legacy))
        elif eng_name == "hnswdb_cured":
            summary["unreachable_mean"] = float(np.mean(unreachable_matrix_cured))
            summary["unreachable_min"] = int(np.min(unreachable_matrix_cured))
            summary["unreachable_max"] = int(np.max(unreachable_matrix_cured))

        results["engines"][eng_name] = summary

    out_path = os.path.join("results", "raw", "seed_matrix_dataset_b.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n[DONE] Saved 5x5 seed matrix results to {out_path}")
    print("\nSummary (Recall@10 at ef=200):")
    for eng_name, s in results["engines"].items():
        unreach_info = f", Unreachable: {s.get('unreachable_mean', 0):.1f}" if "unreachable_mean" in s else ""
        print(f"  {eng_name:15s}: 1T Mean={s['recall_1t_mean']:.4f} (min={s['recall_1t_min']:.4f}, max={s['recall_1t_max']:.4f}), 4T Mean={s['recall_4t_mean']:.4f}{unreach_info}")
        print(f"                   ANOVA Share: Data={s['anova_1t']['share_data_percent']:.1f}%, Build={s['anova_1t']['share_build_percent']:.1f}%, Error={s['anova_1t']['share_error_percent']:.1f}%")

    return results


if __name__ == "__main__":
    run_seed_matrix()
