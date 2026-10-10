#!/usr/bin/env python3
"""
Verify README code examples against live APIs and FastAPI TestClient.
"""

import sys
import os
import json
import re

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "python"))

from fastapi.testclient import TestClient
from hnswdb.server import app, collections


def test_rest_examples():
    """Verify REST curl examples from README against FastAPI TestClient."""
    print("Verifying README REST API examples...")
    client = TestClient(app)
    collections.clear()

    # 1. Create collection
    create_payload = {"dim": 4, "metric": "l2", "max_elements": 1000}
    res = client.post("/collections/products", json=create_payload)
    assert res.status_code == 201, f"Create collection failed: {res.text}"
    assert res.json()["name"] == "products"

    # 2. Insert vector
    insert_payload = {
        "vectors": [[0.1, 0.2, 0.3, 0.4]],
        "ids": [101],
        "metadata": [{"category": "books", "price": 25.0}]
    }
    res = client.post("/collections/products/insert", json=insert_payload)
    assert res.status_code == 200, f"Insert failed: {res.text}"
    assert res.json()["count"] == 1

    # 3. Query collection
    query_payload = {
        "vector": [0.1, 0.2, 0.3, 0.4],
        "k": 5,
        "filter": {"price": {"$lt": 50}}
    }
    res = client.post("/collections/products/query", json=query_payload)
    assert res.status_code == 200, f"Query failed: {res.text}"
    data = res.json()
    assert 101 in data["ids"], f"Expected id 101 in query results, got {data['ids']}"
    assert data["metadata"][0]["category"] == "books"
    print("  REST API examples verified successfully.")


def test_readme_python_snippet():
    """Verify Python snippet from README runs without error."""
    print("Verifying README Python snippet...")
    import numpy as np
    import hnswdb

    dim = 16
    n = 100
    index = hnswdb.HNSWIndex(dim=dim, metric="l2", max_elements=500, M=16, ef_construction=100)

    vectors = np.random.randn(n, dim).astype(np.float32)
    ids = np.arange(n, dtype=np.int64)
    metadata = [{"category": "books" if i % 2 == 0 else "electronics", "price": i} for i in range(n)]

    index.add(vectors, ids, metadata=metadata)

    query = np.random.randn(1, dim).astype(np.float32)
    res_ids, res_dists = index.search(query, k=5, filter={"category": "books", "price": {"$lt": 200}})
    assert len(res_ids[0]) == 5

    tmp_path = os.path.join(PROJECT_ROOT, "results", "raw", "_readme_verify_tmp.hdb")
    index.save(tmp_path)
    loaded = hnswdb.HNSWIndex.load(tmp_path)
    assert len(loaded) == len(index)
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    print("  Python snippet verified successfully.")


def main():
    try:
        test_rest_examples()
        test_readme_python_snippet()
        print("\nPASSED: README verification completed successfully.")
        return 0
    except AssertionError as e:
        print(f"\nFAILED: {e}")
        return 1
    except Exception as e:
        print(f"\nERROR: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
