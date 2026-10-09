import json
import os
import tempfile
import numpy as np
import pytest
from typer.testing import CliRunner
from hnswdb.cli import app

runner = CliRunner()


def test_cli_build_info_query_bench():
    dim = 16
    n = 100
    nq = 10
    rng = np.random.default_rng(42)

    vecs = rng.standard_normal((n, dim)).astype(np.float32)
    queries = rng.standard_normal((nq, dim)).astype(np.float32)

    with tempfile.TemporaryDirectory() as tmpdir:
        vec_path = os.path.join(tmpdir, "data.npy")
        q_path = os.path.join(tmpdir, "queries.npy")
        idx_path = os.path.join(tmpdir, "test.hnsw")
        res_csv = os.path.join(tmpdir, "results.csv")

        np.save(vec_path, vecs)
        np.save(q_path, queries)

        # 1. Build
        res = runner.invoke(app, ["build", "--input", vec_path, "--out", idx_path, "--dim", str(dim), "--metric", "l2", "--M", "16", "--ef-construction", "50"])
        assert res.exit_code == 0
        assert os.path.exists(idx_path)

        # 2. Info
        res = runner.invoke(app, ["info", "--index", idx_path])
        assert res.exit_code == 0
        info_dict = json.loads(res.stdout)
        assert info_dict["size"] == n
        assert info_dict["dim"] == dim

        # 3. Query
        res = runner.invoke(app, ["query", "--index", idx_path, "--query", q_path, "--k", "5", "--ef-search", "30", "--out", res_csv])
        assert res.exit_code == 0
        assert os.path.exists(res_csv)

        # 4. Bench
        res = runner.invoke(app, ["bench", "--index", idx_path, "--queries", q_path, "--k", "5", "--ef-search", "30", "--warmup", "2"])
        assert res.exit_code == 0
        bench_dict = json.loads(res.stdout)
        assert "qps" in bench_dict
        assert "latency_ms" in bench_dict
