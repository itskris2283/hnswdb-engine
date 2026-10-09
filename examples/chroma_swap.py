#!/usr/bin/env python3
"""
HNSWDB as a Drop-in Vector Database Backend
Demonstrates how HNSWDB can replace or power vector database abstractions like ChromaDB or FAISS.
"""

import numpy as np
import hnswdb


class HNSWDBVectorCollection:
    """A lightweight Chroma/Pinecone-like Vector Collection wrapper over HNSWDB."""

    def __init__(self, name: str, dim: int, metric: str = "l2", max_elements: int = 100000):
        self.name = name
        self.dim = dim
        self.metric = metric
        self.index = hnswdb.HNSWIndex(dim=dim, metric=metric, max_elements=max_elements, M=16, ef_construction=100)

    def add(self, ids: list[int], embeddings: np.ndarray, metadatas: list[dict] = None):
        """Add embeddings with IDs and optional metadata."""
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        labels = np.array(ids, dtype=np.int64)
        self.index.add(embeddings, labels, metadata=metadatas)

    def query(self, query_embeddings: np.ndarray, n_results: int = 10, where: dict = None):
        """Query nearest neighbors with optional metadata filter."""
        query_embeddings = np.ascontiguousarray(query_embeddings, dtype=np.float32)
        if query_embeddings.ndim == 1:
            query_embeddings = query_embeddings.reshape(1, -1)

        ids, dists = self.index.search(query_embeddings, k=n_results, filter=where)

        metadatas = []
        for row in ids:
            row_meta = []
            for item_id in row:
                if item_id != -1 and self.index.contains(item_id):
                    row_meta.append(self.index.get_metadata(item_id))
                else:
                    row_meta.append(None)
            metadatas.append(row_meta)

        return {
            "ids": ids.tolist(),
            "distances": dists.tolist(),
            "metadatas": metadatas,
        }

    def delete(self, ids: list[int]):
        """Delete items by ID."""
        return self.index.delete(ids)

    def count(self) -> int:
        return len(self.index)


def main():
    print("=== HNSWDB Drop-in Vector Collection Demo ===")

    collection = HNSWDBVectorCollection(name="user_embeddings", dim=32)

    # 1. Add records
    n = 200
    rng = np.random.default_rng(101)
    vectors = rng.standard_normal((n, 32)).astype(np.float32)
    ids = list(range(n))
    metadatas = [{"tag": "admin" if i % 10 == 0 else "user", "score": int(i * 5)} for i in range(n)]

    collection.add(ids=ids, embeddings=vectors, metadatas=metadatas)
    print(f"Added {collection.count()} vectors into collection '{collection.name}'.")

    # 2. Query with filter
    q_vec = rng.standard_normal((1, 32)).astype(np.float32)
    results = collection.query(query_embeddings=q_vec, n_results=3, where={"tag": "admin"})

    print("\nQuery results (admin only):")
    for doc_id, dist, meta in zip(results["ids"][0], results["distances"][0], results["metadatas"][0]):
        print(f"  ID: {doc_id}, Distance: {dist:.4f}, Metadata: {meta}")

    # 3. Soft-delete and verify
    del_id = results["ids"][0][0]
    print(f"\nDeleting ID {del_id}...")
    collection.delete([del_id])
    print(f"Collection count after delete: {collection.count()}")

    # Re-query
    new_results = collection.query(query_embeddings=q_vec, n_results=3, where={"tag": "admin"})
    print("Query results after deletion:")
    for doc_id, dist in zip(new_results["ids"][0], new_results["distances"][0]):
        print(f"  ID: {doc_id}, Distance: {dist:.4f}")

    print("\n=== Drop-in Collection Demo Completed ===")


if __name__ == "__main__":
    main()
