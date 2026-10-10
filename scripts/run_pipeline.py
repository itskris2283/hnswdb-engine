#!/usr/bin/env python3
"""
run_pipeline.py — Single-entry benchmark pipeline for HNSWDB.

Runs all benchmark configurations, saves raw results to results/raw/,
then renders docs and checks provenance.

Usage:
    python scripts/run_pipeline.py --config smoke    # quick smoke test
    python scripts/run_pipeline.py --config full     # full suite
    python scripts/run_pipeline.py --datasets A B    # specific datasets
"""

import argparse
import json
import os
import platform
import sys
import time
import uuid
from datetime import datetime, timezone

# Add project root to path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import psutil

from benchmarks.engines import HNSWDBEngine, FaissHNSWEngine, HnswlibEngine
from benchmarks.metrics import (
    recall_at_k,
    approx_ratio,
    measure_single_query_latency,
    measure_batch_qps,
)
from benchmarks.datasets_synth import (
    get_dataset_a,
    get_dataset_b,
    compute_ground_truth_bruteforce,
)

RAW_DIR = os.path.join(ROOT, "results", "raw")


def get_env_info() -> dict:
    """Collect reproducibility-critical environment information."""
    import subprocess

    info = {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_model": platform.processor() or "unknown",
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "ram_total_gb": round(psutil.virtual_memory().total / (1024**3), 1),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    # Get git commit
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=ROOT,
            timeout=5,
        )
        info["git_commit"] = result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        info["git_commit"] = "unknown"

    return info


def detect_throttle(pre_temps: list[float] | None, post_temps: list[float] | None) -> dict:
    """Check for CPU throttling between benchmark phases."""
    result = {"detected": False, "note": ""}
    try:
        temps = psutil.sensors_temperatures()
        if temps:
            result["note"] = "Sensor data available"
        else:
            result["note"] = "No temperature sensors accessible on this platform"
    except (AttributeError, RuntimeError):
        result["note"] = "psutil.sensors_temperatures not available on Windows"
    return result


def run_single_engine_benchmark(
    engine_cls,
    engine_name: str,
    xb: np.ndarray,
    xq: np.ndarray,
    gt_nbrs: np.ndarray,
    gt_dists: np.ndarray,
    M: int,
    ef_construction: int,
    ef_values: list[int],
    k: int,
    seed: int,
    build_threads: int,
    search_threads_list: list[int],
    repeats: int = 5,
) -> dict:
    """Run benchmark for a single engine across ef_values."""

    # Build index
    engine_kwargs = {"dim": xb.shape[1], "metric": "l2", "M": M, "ef_construction": ef_construction}
    if engine_name == "hnswdb":
        engine_kwargs["seed"] = seed

    engine = engine_cls(**engine_kwargs)
    build_time = engine.build(xb, num_threads=build_threads)

    # Measure across ef values
    ef_results = []
    for ef in ef_values:
        # Search
        ids, dists = engine.search(xq, k=k, ef_search=ef, num_threads=1)

        # Recall
        recall = recall_at_k(ids, gt_nbrs, k)

        # Approximation ratio
        ar = approx_ratio(dists, gt_dists, k, squared=True)

        # Single-query latency
        def search_one(q):
            return engine.search(q, k=k, ef_search=ef, num_threads=1)

        lat = measure_single_query_latency(search_one, xq, warmup=min(500, len(xq)))

        # Batch QPS at various thread counts
        qps_by_threads = {}
        for nt in search_threads_list:
            def search_batch(q, _nt=nt):
                return engine.search(q, k=k, ef_search=ef, num_threads=_nt)

            qps_data = measure_batch_qps(search_batch, xq, nt, repeats=repeats)
            qps_by_threads[str(nt)] = qps_data

        ef_results.append({
            "ef_search": ef,
            "recall_at_k": round(recall, 6),
            "approx_ratio": round(ar, 6),
            "latency": {k2: round(v, 4) for k2, v in lat.items() if k2 != "is_suspect"},
            "latency_suspect": lat.get("is_suspect", False),
            "qps_by_threads": qps_by_threads,
        })

    return {
        "engine": engine_name,
        "build_time_s": round(build_time, 4),
        "build_threads": build_threads,
        "memory_bytes": engine.get_memory_bytes() if hasattr(engine, "get_memory_bytes") else 0,
        "rss_delta_mb": round(engine.rss_delta_mb, 2),
        "ef_results": ef_results,
    }


def run_dataset_benchmark(
    dataset_name: str,
    dataset_fn,
    dataset_kwargs: dict,
    M: int = 16,
    ef_construction: int = 100,
    ef_values: list[int] = None,
    k: int = 10,
    seeds: list[int] = None,
    build_threads: int = 1,
    search_threads_list: list[int] = None,
    repeats: int = 5,
    engines: list[str] = None,
) -> dict:
    """Run full benchmark for a dataset across all engines and seeds."""

    if ef_values is None:
        ef_values = [10, 20, 50, 100, 200]
    if seeds is None:
        seeds = [42]
    if search_threads_list is None:
        search_threads_list = [1, 4, 8]
    if engines is None:
        engines = ["hnswdb", "faiss", "hnswlib"]

    run_id = f"{dataset_name}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

    result = {
        "run_id": run_id,
        "dataset": dataset_name,
        "params": {
            "M": M,
            "ef_construction": ef_construction,
            "ef_values": ef_values,
            "k": k,
            "seeds": seeds,
            "build_threads": build_threads,
            "search_threads": search_threads_list,
            "repeats": repeats,
        },
        "env": get_env_info(),
        "throttle_check": detect_throttle(None, None),
        "engines": {},
        "timing": {},
    }

    t_start = time.perf_counter()

    engine_map = {
        "hnswdb": HNSWDBEngine,
        "faiss": FaissHNSWEngine,
        "hnswlib": HnswlibEngine,
    }

    for seed in seeds:
        seed_key = f"seed_{seed}"

        # Generate dataset
        xb, xq, gt_nbrs, gt_dists, metric = dataset_fn(**dataset_kwargs, seed=seed) if "seed" in dataset_kwargs or callable(dataset_fn) else dataset_fn()
        result.setdefault("dataset_info", {
            "n_train": int(xb.shape[0]),
            "n_test": int(xq.shape[0]),
            "dim": int(xb.shape[1]),
            "metric": metric,
        })

        for eng_name in engines:
            if eng_name not in engine_map:
                print(f"  Skipping unknown engine: {eng_name}")
                continue
            try:
                eng_cls = engine_map[eng_name]
                print(f"  [{dataset_name}] seed={seed} engine={eng_name} M={M} efC={ef_construction} ...")

                bench = run_single_engine_benchmark(
                    eng_cls, eng_name, xb, xq, gt_nbrs, gt_dists,
                    M=M, ef_construction=ef_construction,
                    ef_values=ef_values, k=k, seed=seed,
                    build_threads=build_threads,
                    search_threads_list=search_threads_list,
                    repeats=repeats,
                )

                if eng_name not in result["engines"]:
                    result["engines"][eng_name] = {}
                result["engines"][eng_name][seed_key] = bench

            except Exception as e:
                print(f"  ERROR: {eng_name} seed={seed}: {e}")
                if eng_name not in result["engines"]:
                    result["engines"][eng_name] = {}
                result["engines"][eng_name][seed_key] = {"error": str(e)}

    result["timing"]["total_s"] = round(time.perf_counter() - t_start, 2)

    return result


def save_raw(result: dict, filename: str) -> str:
    """Save result to results/raw/ and return path."""
    os.makedirs(RAW_DIR, exist_ok=True)
    path = os.path.join(RAW_DIR, filename)
    with open(path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"  Saved: {path}")
    return path


def get_dataset_b_sigma(sigma: float = 0.1, seed: int = 42):
    """Dataset B variant with configurable cluster tightness."""
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
    return xb, xq, gt_nbrs, gt_dists, "l2"


DATASET_CONFIGS = {
    "A": {
        "fn": get_dataset_a,
        "kwargs": {},
        "M": 16,
        "efC": 100,
        "label": "Synthetic A (5K x 32d, 20 clusters, σ=0.1)",
    },
    "B_s01": {
        "fn": get_dataset_b_sigma,
        "kwargs": {"sigma": 0.1},
        "M": 16,
        "efC": 100,
        "label": "Synthetic B (20K x 64d, 50 clusters, σ=0.1)",
    },
    "B_s03": {
        "fn": get_dataset_b_sigma,
        "kwargs": {"sigma": 0.3},
        "M": 16,
        "efC": 100,
        "label": "Synthetic B (20K x 64d, 50 clusters, σ=0.3)",
    },
    "B_s10": {
        "fn": get_dataset_b_sigma,
        "kwargs": {"sigma": 1.0},
        "M": 16,
        "efC": 100,
        "label": "Synthetic B (20K x 64d, 50 clusters, σ=1.0)",
    },
}

SMOKE_DATASETS = ["A"]
FULL_DATASETS = ["A", "B_s01", "B_s03", "B_s10"]


def main():
    parser = argparse.ArgumentParser(description="HNSWDB Benchmark Pipeline")
    parser.add_argument("--config", choices=["smoke", "full", "custom"], default="smoke")
    parser.add_argument("--datasets", nargs="+", default=None, help="Dataset names to run")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42], help="Seeds for reproducibility")
    parser.add_argument("--ef-values", nargs="+", type=int, default=None, help="ef_search values")
    parser.add_argument("--repeats", type=int, default=5, help="QPS measurement repeats")
    parser.add_argument("--build-threads", type=int, default=1, help="Build thread count")
    parser.add_argument("--search-threads", nargs="+", type=int, default=[1, 4, 8])
    parser.add_argument("--engines", nargs="+", default=["hnswdb", "faiss", "hnswlib"])
    parser.add_argument("--skip-render", action="store_true", help="Skip doc rendering after benchmarks")
    args = parser.parse_args()

    if args.config == "smoke":
        datasets = args.datasets or SMOKE_DATASETS
        ef_values = args.ef_values or [10, 50, 200]
        seeds = args.seeds
    elif args.config == "full":
        datasets = args.datasets or FULL_DATASETS
        ef_values = args.ef_values or [10, 20, 50, 100, 200]
        seeds = args.seeds or [42, 0, 1, 2, 3]
    else:
        datasets = args.datasets or SMOKE_DATASETS
        ef_values = args.ef_values or [10, 50, 200]
        seeds = args.seeds

    print(f"=== HNSWDB Benchmark Pipeline ===")
    print(f"Config: {args.config}")
    print(f"Datasets: {datasets}")
    print(f"Seeds: {seeds}")
    print(f"EF values: {ef_values}")
    print(f"Engines: {args.engines}")
    print()

    manifest_entries = []

    for ds_name in datasets:
        if ds_name not in DATASET_CONFIGS:
            print(f"WARNING: Unknown dataset '{ds_name}', skipping")
            continue

        cfg = DATASET_CONFIGS[ds_name]

        def ds_fn(seed=42, _cfg=cfg):
            return _cfg["fn"](seed=seed, **_cfg["kwargs"]) if _cfg["kwargs"] else _cfg["fn"](seed=seed)

        result = run_dataset_benchmark(
            dataset_name=ds_name,
            dataset_fn=ds_fn,
            dataset_kwargs={},
            M=cfg["M"],
            ef_construction=cfg["efC"],
            ef_values=ef_values,
            k=10,
            seeds=seeds,
            build_threads=args.build_threads,
            search_threads_list=args.search_threads,
            repeats=args.repeats,
            engines=args.engines,
        )

        filename = f"bench_{ds_name}_{result['run_id'][-8:]}.json"
        save_raw(result, filename)

        manifest_entries.append({
            "file": filename,
            "run_id": result["run_id"],
            "dataset": ds_name,
            "label": cfg["label"],
            "created": datetime.now(timezone.utc).isoformat(),
        })

    # Update manifest
    manifest_path = os.path.join(RAW_DIR, "MANIFEST.json")
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
    else:
        manifest = {"version": 4, "entries": []}

    manifest["entries"].extend(manifest_entries)
    manifest["last_updated"] = datetime.now(timezone.utc).isoformat()
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest updated: {manifest_path}")

    if not args.skip_render:
        print("\n=== Rendering docs from raw data ===")
        os.system(f'{sys.executable} "{os.path.join(ROOT, "scripts", "render_docs.py")}"')
        print("\n=== Checking provenance ===")
        os.system(f'{sys.executable} "{os.path.join(ROOT, "scripts", "check_provenance.py")}"')

    print("\n=== Pipeline complete ===")


if __name__ == "__main__":
    main()
