import os
import sys
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import hnswdb
import json
import pytest


def test_benchmark_smoke():
    from benchmarks.run_benchmark import run_benchmark
    from benchmarks.make_plots import generate_plots
    from benchmarks.make_readme_tables import generate_tables

    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "benchmarks", "configs", "smoke.json")
    assert os.path.exists(config_path)

    df = run_benchmark(config_path)
    assert not df.empty
    assert "recall_at_10" in df.columns
    assert "latency_p50_ms" in df.columns
    assert "sequential_qps" in df.columns
    assert "batch_qps_1th" in df.columns

    # Verify HNSWDB results exist and have valid recall
    hnsw_rows = df[df["engine"] == "hnswdb"]
    assert not hnsw_rows.empty
    max_recall = hnsw_rows["recall_at_10"].max()
    assert max_recall >= 0.90

    # Generate plots and tables
    generate_plots()
    tables_md = generate_tables()
    assert "hnswdb" in tables_md
