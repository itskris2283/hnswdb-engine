import os
import tempfile
import numpy as np
import pytest
from fastapi.testclient import TestClient
from hnswdb.server import app, collections


@pytest.fixture(autouse=True)
def clear_collections():
    collections.clear()
    yield
    collections.clear()


client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_full_rest_workflow():
    dim = 8
    col_name = "test_col"

    # 1. Create collection
    res = client.post(
        f"/collections/{col_name}",
        json={"dim": dim, "metric": "l2", "max_elements": 100, "M": 16, "ef_construction": 50},
    )
    assert res.status_code == 201
    assert res.json()["name"] == col_name

    # 2. Insert vectors
    n = 10
    vecs = np.random.randn(n, dim).astype(np.float32).tolist()
    ids = list(range(n))
    metadata = [{"name": f"item_{i}", "price": i * 10} for i in range(n)]

    res = client.post(
        f"/collections/{col_name}/insert",
        json={"vectors": vecs, "ids": ids, "metadata": metadata},
    )
    assert res.status_code == 200
    assert res.json()["count"] == n

    # 3. Query
    q_vec = vecs[0]
    res = client.post(
        f"/collections/{col_name}/query",
        json={"vector": q_vec, "k": 3, "ef_search": 20},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["ids"][0] == 0
    assert body["distances"][0] == pytest.approx(0.0, abs=1e-5)
    assert body["metadata"][0]["name"] == "item_0"
    assert "took_ms" in body

    # 4. Filtered Query
    res = client.post(
        f"/collections/{col_name}/query",
        json={"vector": q_vec, "k": 5, "filter": {"price": {"$gte": 50}}},
    )
    assert res.status_code == 200
    for fid, meta in zip(res.json()["ids"], res.json()["metadata"]):
        if fid != -1:
            assert meta["price"] >= 50

    # 5. Stats
    res = client.get(f"/collections/{col_name}/stats")
    assert res.status_code == 200
    assert res.json()["size"] == n

    # 6. Delete vectors
    res = client.request(
        "DELETE",
        f"/collections/{col_name}/vectors",
        json={"ids": [0, 1]},
    )
    assert res.status_code == 200
    assert res.json()["deleted_count"] == 2

    # 7. Save and Load
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "col.hnsw")
        res = client.post(f"/collections/{col_name}/save?path={save_path}")
        assert res.status_code == 200

        # Load into new collection name
        res = client.post(f"/collections/col_loaded/load?path={save_path}")
        assert res.status_code == 200
        assert res.json()["size"] == n - 2
