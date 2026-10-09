import os
import sys
import argparse
import datetime
import json
import time
import subprocess

# On Windows, import faiss and hnswlib before any local C++ extensions to prevent OpenMP runtime collision
try:
    import faiss
except ImportError:
    faiss = None

try:
    import hnswlib
except ImportError:
    hnswlib = None

import numpy as np
import pandas as pd

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import hnswdb
from benchmarks.env_info import get_env_info
from benchmarks.download_datasets import load_dataset
from benchmarks.engines import HNSWDBEngine, FaissHNSWEngine, HnswlibEngine, FaissFlatEngine
from benchmarks.metrics import (
    recall_at_k,
    approx_ratio,
    measure_single_query_latency,
    measure_batch_qps,
)
from benchmarks.equal_recall import interpolate_equal_recall


RESULTS_DIR = os.path.join(project_root, "results")


def benchmark_single_engine_worker(ds_name: str, engine_name: str, M: int, ef_c: int, ef_s_list: list, k_list: list, threads_list: list, seed: int = 42) -> list:
    """Worker function executed per engine to evaluate performance and accuracy."""
    train_data, test_data, gt_neighbors, gt_distances, metric = load_dataset(ds_name)
    n_train, dim = train_data.shape
    n_test = test_data.shape[0]

    max_k = max(k_list) if isinstance(k_list, list) else int(k_list)

    # Compute or verify ground truth
    if gt_neighbors is None or gt_distances is None:
        flat_engine = FaissFlatEngine(dim=dim, metric=metric)
        flat_engine.build(train_data)
        gt_neighbors, gt_distances = flat_engine.search(test_data, k=max_k)
    else:
        gt_neighbors = gt_neighbors[:, :max_k]
        gt_distances = gt_distances[:, :max_k]

    # Instantiate engine
    if engine_name == "hnswdb":
        engine = HNSWDBEngine(dim=dim, metric=metric, M=M, ef_construction=ef_c, seed=seed)
    elif engine_name == "faiss_hnsw":
        engine = FaissHNSWEngine(dim=dim, metric=metric, M=M, ef_construction=ef_c)
    elif engine_name == "hnswlib":
        engine = HnswlibEngine(dim=dim, metric=metric, M=M, ef_construction=ef_c)
    else:
        raise ValueError(f"Unknown engine: {engine_name}")

    build_time = engine.build(train_data, num_threads=0)
    disk_bytes = engine.get_disk_bytes()
    rss_delta_mb = engine.rss_delta_mb

    records = []
    primary_k = 10 if 10 in (k_list if isinstance(k_list, list) else [k_list]) else max_k
    eval_queries = test_data[:1000] if n_test > 1000 else test_data

    for ef_s in ef_s_list:
        # 1. Single-query latency (strictly 1 query per call, 1 thread)
        lat_stats = measure_single_query_latency(
            lambda q: engine.search(q, k=primary_k, ef_search=ef_s, num_threads=1),
            eval_queries,
            warmup=min(100, len(eval_queries)),
        )

        # 2. Accuracy search on full test set with 1 thread
        found_ids, found_dists = engine.search(test_data, k=max_k, ef_search=ef_s, num_threads=1)

        recalls = {}
        target_ks = [1, 10, 100] if isinstance(k_list, list) else [k_list]
        for tk in target_ks:
            if tk <= max_k:
                recalls[f"recall_at_{tk}"] = round(recall_at_k(found_ids, gt_neighbors, tk), 4)

        approx = round(approx_ratio(found_dists, gt_distances, primary_k, squared=(metric == "l2")), 4)

        # 3. Batch QPS at different thread counts
        batch_qps_metrics = {}
        for num_th in threads_list:
            qps_res = measure_batch_qps(
                lambda queries, th: engine.search(queries, k=primary_k, ef_search=ef_s, num_threads=th),
                eval_queries,
                num_threads=num_th,
                repeats=3,
                warmup=min(100, len(eval_queries)),
            )
            batch_qps_metrics[f"batch_qps_{num_th}th"] = round(qps_res["median"], 2)
            batch_qps_metrics[f"batch_qps_{num_th}th_min"] = round(qps_res["min"], 2)
            batch_qps_metrics[f"batch_qps_{num_th}th_max"] = round(qps_res["max"], 2)

        # 4. Pure C++ benchmark search latency (for HNSWDB only)
        cpp_lat_p50 = None
        if hasattr(engine, "bench_search_latency"):
            try:
                cpp_lat_p50 = round(float(engine.bench_search_latency(test_data, k=primary_k, ef_search=ef_s)), 4)
            except Exception:
                pass

        rec = {
            "dataset": ds_name,
            "n_train": int(n_train),
            "dim": int(dim),
            "metric": metric,
            "engine": engine_name,
            "M": int(M),
            "ef_construction": int(ef_c),
            "ef_search": int(ef_s),
            "k": int(primary_k),
            "build_time_s": round(float(build_time), 3),
            "disk_mb": round(float(disk_bytes) / (1024.0 * 1024.0), 3),
            "rss_delta_mb": round(float(rss_delta_mb), 2),
            "approx_ratio": approx,
            "latency_p50_ms": round(float(lat_stats["p50"]), 4),
            "latency_p95_ms": round(float(lat_stats["p95"]), 4),
            "latency_p99_ms": round(float(lat_stats["p99"]), 4),
            "latency_p999_ms": round(float(lat_stats["p999"]), 4),
            "latency_mean_ms": round(float(lat_stats["mean"]), 4),
            "latency_max_ms": round(float(lat_stats["max"]), 4),
            "sequential_qps": round(float(lat_stats["sequential_qps"]), 2),
            "is_suspect": lat_stats["is_suspect"],
            "cpp_latency_p50_ms": cpp_lat_p50,
        }
        rec.update(recalls)
        rec.update(batch_qps_metrics)
        records.append(rec)

    return records


def run_benchmark(config_path: str):
    with open(config_path, "r") as f:
        config = json.load(f)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = config.get("name", "bench")

    datasets = config.get("datasets", [config.get("dataset", "synthetic_5000_32_l2")])
    engines_list = config.get("engines", ["hnswdb", "faiss_hnsw", "hnswlib"])
    m_list = config.get("M", 16)
    if isinstance(m_list, int):
        m_list = [m_list]
    ef_c_list = config.get("ef_construction", 100)
    if isinstance(ef_c_list, int):
        ef_c_list = [ef_c_list]
    ef_s_list = config.get("ef_search", [10, 20, 50, 100])
    if isinstance(ef_s_list, int):
        ef_s_list = [ef_s_list]
    k_list = config.get("k", [10])
    if isinstance(k_list, int):
        k_list = [k_list]
    threads_list = config.get("threads", [1, 4])
    seed = config.get("seed", 42)

    env_info = get_env_info()
    all_records = []

    for ds_name in datasets:
        print(f"\n==========================================")
        print(f"Dataset: {ds_name}")
        print(f"==========================================")

        for M in m_list:
            for ef_c in ef_c_list:
                print(f"\n--- Testing Index Params: M={M}, ef_construction={ef_c} ---")

                for engine_name in engines_list:
                    print(f"\nEvaluating Engine: {engine_name}...")
                    engine_records = benchmark_single_engine_worker(
                        ds_name=ds_name,
                        engine_name=engine_name,
                        M=M,
                        ef_c=ef_c,
                        ef_s_list=ef_s_list,
                        k_list=k_list,
                        threads_list=threads_list,
                        seed=seed,
                    )
                    for rec in engine_records:
                        rec["timestamp"] = timestamp
                        all_records.append(rec)
                        r10 = rec.get("recall_at_10", 0.0)
                        p50 = rec["latency_p50_ms"]
                        seq_qps = rec["sequential_qps"]
                        b1 = rec.get("batch_qps_1th", 0)
                        b4 = rec.get("batch_qps_4th", 0)
                        print(f"    ef={rec['ef_search']:3d} | R@10: {r10:.4f} | p50: {p50:.3f}ms | seqQPS: {seq_qps:.0f} | batchQPS: 1th={b1:.0f}, 4th={b4:.0f}")

    # Interpolate equal recall targets
    equal_recall_records = interpolate_equal_recall(all_records, targets=[0.90, 0.95, 0.99])

    full_output = {
        "env_info": env_info,
        "config": config,
        "records": all_records,
        "equal_recall_interpolations": equal_recall_records,
    }

    # Save results
    df = pd.DataFrame(all_records)
    csv_path = os.path.join(RESULTS_DIR, f"{run_name}_{timestamp}.csv")
    json_path = os.path.join(RESULTS_DIR, f"{run_name}_{timestamp}.json")
    latest_csv = os.path.join(RESULTS_DIR, "latest.csv")
    latest_json = os.path.join(RESULTS_DIR, "latest.json")

    df.to_csv(csv_path, index=False)
    df.to_csv(latest_csv, index=False)

    with open(json_path, "w") as f:
        json.dump(full_output, f, indent=2)
    with open(latest_json, "w") as f:
        json.dump(full_output, f, indent=2)

    print(f"\n==========================================")
    print(f"Results saved to:")
    print(f"  {csv_path}")
    print(f"  {json_path}")
    print(f"==========================================")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run HNSWDB ANN Benchmarks")
    parser.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "configs", "smoke.json"), help="Config JSON path")
    args = parser.parse_args()
    run_benchmark(args.config)
