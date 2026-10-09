import time
import numpy as np


def per_query_recall(found: np.ndarray, truth: np.ndarray, k: int) -> np.ndarray:
    """Compute per-query recall array for k neighbors."""
    if len(found) == 0 or len(truth) == 0 or k <= 0:
        return np.array([], dtype=np.float64)
    recalls = np.zeros(len(found), dtype=np.float64)
    for i, (f, t) in enumerate(zip(found, truth)):
        f_k = set(f[:k].tolist())
        t_k = set(t[:k].tolist())
        f_k.discard(-1)
        t_k.discard(-1)
        denom = float(min(k, len(t_k)))
        if denom > 0:
            recalls[i] = len(f_k & t_k) / denom
        else:
            recalls[i] = 1.0
    return recalls


def recall_at_k(found: np.ndarray, truth: np.ndarray, k: int) -> float:
    """Compute mean Recall@k between found IDs and ground truth IDs."""
    arr = per_query_recall(found, truth, k)
    return float(np.mean(arr)) if len(arr) > 0 else 0.0


def self_retrieval_rate(search_fn, xb: np.ndarray, labels: np.ndarray = None) -> float:
    """
    Check 1-NN self-retrieval: querying with xb[i] should return label[i] at rank 0.
    search_fn(query_vector_1d_or_2d) -> (ids, dists)
    """
    n = len(xb)
    if n == 0:
        return 1.0
    if labels is None:
        labels = np.arange(n, dtype=np.int64)
    correct = 0
    for i in range(n):
        q = np.ascontiguousarray(xb[i:i+1], dtype=np.float32)
        ids, _ = search_fn(q)
        if len(ids) > 0 and len(ids[0]) > 0 and ids[0][0] == labels[i]:
            correct += 1
    return float(correct / n)



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
