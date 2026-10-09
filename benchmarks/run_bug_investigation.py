"""
Script to benchmark and collect diagnostics for duplicate-neighbor bug investigation.
Outputs results to results/raw/bug_<state>_<dataset>.json
"""

import sys
import os
import json
import time
import argparse
import subprocess
import numpy as np
import hnswdb
from hnswdb import HNSWIndex
from benchmarks.datasets_synth import get_dataset_a, get_dataset_b, get_dataset_c
from benchmarks.metrics import recall_at_k, self_retrieval_rate
from benchmarks.env_info import get_env_info



def get_git_commit():
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return r.stdout.strip()
    except Exception:
        return "unknown"


def run_eval(dataset_name: str, state: str, out_dir: str):
    print(f"--- Running Bug Investigation: Dataset={dataset_name}, State={state} ---")
    if dataset_name == "dataset_a":
        xb, xq, gt_nbrs, gt_dists, metric = get_dataset_a()
    elif dataset_name == "dataset_b":
        xb, xq, gt_nbrs, gt_dists, metric = get_dataset_b()
    elif dataset_name == "dataset_c" or dataset_name == "fashion_mnist":
        xb, xq, gt_nbrs, gt_dists, metric = get_dataset_c()
        dataset_name = "fashion_mnist"
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    dim = xb.shape[1]
    n_train = len(xb)
    M = 16
    ef_c = 100
    seed = 42

    idx = HNSWIndex(dim=dim, metric=metric, max_elements=n_train, M=M, ef_construction=ef_c, ef_search=50, seed=seed)
    
    t0 = time.perf_counter()
    idx.add(xb, np.arange(n_train, dtype=np.int64))
    build_time_s = time.perf_counter() - t0

    diag = idx.diagnostics()
    print("Diagnostics:", diag)

    # Search at various ef values: 10, 20, 50, 100, 200
    ef_list = [10, 20, 50, 100, 200]
    recall_results = {}
    for ef in ef_list:
        idx.set_ef_search(ef)
        ids, dists = idx.search(xq, k=10)
        rec = recall_at_k(ids, gt_nbrs, k=10)
        recall_results[f"recall_at_10_ef_{ef}"] = round(rec, 4)
        print(f"  ef={ef}: recall@10 = {rec:.4f}")

    # Self-retrieval
    idx.set_ef_search(50)
    self_ret = self_retrieval_rate(lambda q: idx.search(q, k=1), xb[:min(1000, n_train)])
    print(f"  Self-retrieval rate (1-NN): {self_ret:.4f}")

    env = get_env_info()
    commit = get_git_commit()


    result_data = {
        "dataset": dataset_name,
        "state": state,
        "commit": commit,
        "n_train": n_train,
        "dim": dim,
        "M": M,
        "ef_construction": ef_c,
        "seed": seed,
        "build_time_s": round(build_time_s, 4),
        "diagnostics": diag,
        "recalls": recall_results,
        "self_retrieval_rate": round(self_ret, 4),
        "env": env,
    }

    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"bug_{state}_{dataset_name}.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=2)
    print(f"Saved: {out_file}\n")
    return out_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["dataset_a", "dataset_b", "fashion_mnist", "all"], default="all")
    parser.add_argument("--state", choices=["before", "after"], required=True)
    parser.add_argument("--out-dir", default=os.path.join("results", "raw"))
    args = parser.parse_args()

    datasets = ["dataset_a", "dataset_b", "fashion_mnist"] if args.dataset == "all" else [args.dataset]
    for ds in datasets:
        run_eval(ds, args.state, args.out_dir)
