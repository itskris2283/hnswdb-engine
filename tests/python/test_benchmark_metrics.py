import pytest
import numpy as np
from benchmarks.metrics import (
    recall_at_k,
    approx_ratio,
    measure_single_query_latency,
    measure_batch_qps,
)
from benchmarks.equal_recall import interpolate_equal_recall


def test_recall_at_k_exact_and_disjoint():
    found = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    truth = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    assert recall_at_k(found, truth, 3) == pytest.approx(1.0)
    assert recall_at_k(found, truth, 1) == pytest.approx(1.0)

    # Disjoint
    disjoint = np.array([[10, 11, 12], [13, 14, 15], [16, 17, 18]])
    assert recall_at_k(disjoint, truth, 3) == pytest.approx(0.0)

    # Partial overlap (2 out of 3)
    partial = np.array([[1, 2, 99], [4, 5, 99], [7, 8, 99]])
    assert recall_at_k(partial, truth, 3) == pytest.approx(2.0 / 3.0)


def test_approx_ratio_metrics():
    # Exact distances
    true_d = np.array([[1.0, 4.0, 9.0], [1.0, 4.0, 9.0]], dtype=np.float32)
    found_d = np.array([[1.0, 4.0, 9.0], [1.0, 4.0, 9.0]], dtype=np.float32)
    # L2 with sqrt
    assert approx_ratio(found_d, true_d, 3, squared=True) == pytest.approx(1.0)

    # 2x distance squared -> 4x
    found_d_larger = np.array([[4.0, 16.0, 36.0], [4.0, 16.0, 36.0]], dtype=np.float32)
    assert approx_ratio(found_d_larger, true_d, 3, squared=True) == pytest.approx(2.0)


def test_measure_single_query_latency_and_suspect_validation():
    xq = np.random.randn(50, 16).astype(np.float32)
    dummy_search = lambda q: (np.zeros((1, 10), dtype=np.int64), np.zeros((1, 10), dtype=np.float32))

    res = measure_single_query_latency(dummy_search, xq, warmup=5)
    assert "p50" in res
    assert "p95" in res
    assert "p99" in res
    assert "p999" in res
    assert "sequential_qps" in res
    assert "is_suspect" in res
    assert res["is_suspect"] is False
    assert res["sequential_qps"] > 0


def test_measure_batch_qps_repeats():
    xq = np.random.randn(100, 16).astype(np.float32)
    dummy_batch = lambda q, th: (np.zeros((len(q), 10), dtype=np.int64), np.zeros((len(q), 10), dtype=np.float32))

    qps_res = measure_batch_qps(dummy_batch, xq, num_threads=1, repeats=5, warmup=10)
    assert "median" in qps_res
    assert "min" in qps_res
    assert "max" in qps_res
    assert len(qps_res["runs"]) == 5
    assert qps_res["median"] > 0
    assert qps_res["min"] <= qps_res["median"] <= qps_res["max"]


def test_equal_recall_interpolation():
    records = [
        {"dataset": "test_ds", "engine": "hnswdb", "M": 16, "ef_construction": 100, "ef_search": 10, "recall_at_10": 0.88, "latency_p50_ms": 0.05, "sequential_qps": 20000.0, "batch_qps_1th": 25000.0},
        {"dataset": "test_ds", "engine": "hnswdb", "M": 16, "ef_construction": 100, "ef_search": 20, "recall_at_10": 0.96, "latency_p50_ms": 0.08, "sequential_qps": 12000.0, "batch_qps_1th": 15000.0},
        {"dataset": "test_ds", "engine": "hnswdb", "M": 16, "ef_construction": 100, "ef_search": 50, "recall_at_10": 0.995, "latency_p50_ms": 0.15, "sequential_qps": 6500.0, "batch_qps_1th": 8000.0},
    ]

    interpolated = interpolate_equal_recall(records, targets=[0.90, 0.95, 0.99])
    assert len(interpolated) == 3

    # Check target 0.90
    t90 = next(r for r in interpolated if r["target_recall"] == 0.90)
    assert t90["status"] == "achieved"
    assert 0.05 <= t90["interpolated_p50_ms"] <= 0.08
    assert 12000.0 <= t90["interpolated_sequential_qps"] <= 20000.0

    # Check unreachable target
    interpolated_unreachable = interpolate_equal_recall(records, targets=[0.999])
    assert interpolated_unreachable[0]["status"] == "unreachable"
