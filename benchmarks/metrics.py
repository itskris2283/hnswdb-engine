import time
import numpy as np


def recall_at_k(found: np.ndarray, truth: np.ndarray, k: int) -> float:
    """Compute mean Recall@k between found IDs and ground truth IDs."""
    if len(found) == 0 or len(truth) == 0 or k <= 0:
        return 0.0
    recalls = []
    for f, t in zip(found, truth):
        f_k = set(f[:k].tolist())
        t_k = set(t[:k].tolist())
        # Remove padding / negative IDs
        f_k.discard(-1)
        t_k.discard(-1)
        if len(t_k) == 0:
            continue
        recalls.append(len(f_k & t_k) / float(min(k, len(t_k))))
    return float(np.mean(recalls)) if recalls else 0.0


def approx_ratio(found_d: np.ndarray, true_d: np.ndarray, k: int, squared: bool = True) -> float:
    """
    Compute true approximation ratio: mean(d_approx / d_exact).
    For L2 (squared=True), converts squared distances to Euclidean distances.
    """
    if len(found_d) == 0 or len(true_d) == 0 or k <= 0:
        return 1.0
    f = np.sqrt(np.maximum(found_d[:, :k], 0)) if squared else found_d[:, :k]
    t = np.sqrt(np.maximum(true_d[:, :k], 0)) if squared else true_d[:, :k]
    ratios = f / np.maximum(t, 1e-12)
    # Clip any NaN or inf resulting from missing results
    ratios = np.nan_to_num(ratios, nan=1.0, posinf=2.0, neginf=1.0)
    return float(np.mean(ratios))


def measure_single_query_latency(search_one_fn, xq: np.ndarray, warmup: int = 500) -> dict:
    """
    Measure single-query latency with warm-up.
    Guarantees that each query is passed individually.
    Returns percentile latency (in milliseconds) and sequential QPS.
    """
    n_queries = len(xq)
    warmup_n = min(warmup, n_queries)
    for i in range(warmup_n):
        q = np.ascontiguousarray(xq[i:i+1], dtype=np.float32)
        search_one_fn(q)

    latencies_ms = np.empty(n_queries, dtype=np.float64)
    for i in range(n_queries):
        q = np.ascontiguousarray(xq[i:i+1], dtype=np.float32)
        t0 = time.perf_counter_ns()
        search_one_fn(q)
        latencies_ms[i] = (time.perf_counter_ns() - t0) / 1e6

    total_time_s = np.sum(latencies_ms) / 1000.0
    sequential_qps = float(n_queries / total_time_s) if total_time_s > 0 else 0.0
    mean_lat_ms = float(np.mean(latencies_ms))

    # Suspect row check
    implied_qps = (1000.0 / mean_lat_ms) if mean_lat_ms > 0 else 0.0
    rel_diff = abs(implied_qps - sequential_qps) / max(sequential_qps, 1e-9)
    is_suspect = bool(rel_diff > 0.05)

    return {
        "p50": float(np.percentile(latencies_ms, 50)),
        "p95": float(np.percentile(latencies_ms, 95)),
        "p99": float(np.percentile(latencies_ms, 99)),
        "p999": float(np.percentile(latencies_ms, 99.9)),
        "mean": mean_lat_ms,
        "max": float(np.max(latencies_ms)),
        "sequential_qps": sequential_qps,
        "is_suspect": is_suspect,
    }


def measure_batch_qps(search_batch_fn, xq: np.ndarray, num_threads: int, repeats: int = 5, warmup: int = 500) -> dict:
    """
    Measure multi-threaded batch throughput (QPS).
    Warms up, then runs repeats to record min, median, max QPS.
    """
    n_queries = len(xq)
    # Warmup
    warmup_n = min(warmup, n_queries)
    if warmup_n > 0:
        search_batch_fn(np.ascontiguousarray(xq[:warmup_n], dtype=np.float32), num_threads)

    qps_runs = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        search_batch_fn(xq, num_threads)
        elapsed = time.perf_counter() - t0
        if elapsed > 0:
            qps_runs.append(n_queries / elapsed)
        else:
            qps_runs.append(0.0)

    return {
        "median": float(np.median(qps_runs)),
        "min": float(np.min(qps_runs)),
        "max": float(np.max(qps_runs)),
        "runs": [round(float(r), 2) for r in qps_runs],
    }


# Backwards compatible alias
measure_latency = measure_single_query_latency
measure_qps = lambda fn, xq, threads, repeats=5: measure_batch_qps(fn, xq, threads, repeats)["median"]
