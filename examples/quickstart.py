#!/usr/bin/env python3
"""
HNSWDB Quickstart Example
Demonstrates creating an HNSW index, inserting vectors with metadata,
performing approximate nearest neighbor search with metadata filtering,
and saving/loading the index using .hdb persistence.
"""

import os
import numpy as np
import hnswdb


def main():
    print("=== HNSWDB Quickstart Demo ===")

    # 1. Initialize Index
    dim = 64
    max_elements = 1000
    print(f"\n1. Creating HNSW index (dim={dim}, metric=l2, M=16, ef_construction=100)...")
    index = hnswdb.HNSWIndex(
        dim=dim,
        metric="l2",
        max_elements=max_elements,
        M=16,
        ef_construction=100,
        ef_search=50,
        seed=42,
    )

    # 2. Generate sample vectors and metadata
    n_vectors = 500
    rng = np.random.default_rng(42)
    vectors = rng.standard_normal((n_vectors, dim)).astype(np.float32)
    labels = np.arange(n_vectors, dtype=np.int64)

    categories = ["electronics", "clothing", "home", "books", "garden"]
    metadata = [
        {
            "category": categories[i % len(categories)],
            "price": float(10 + (i * 3) % 200),
            "in_stock": (i % 3 != 0),
            "rating": float(3.0 + (i % 20) / 10.0),
        }
        for i in range(n_vectors)
    ]

    print(f"2. Adding {n_vectors} vectors with rich metadata...")
    index.add(vectors, labels, metadata=metadata)
    print(f"   Index size: {len(index)} active elements")

    # 3. Standard Vector Search
    query = rng.standard_normal((1, dim)).astype(np.float32)
    k = 5
    print(f"\n3. Performing standard ANN search (top-{k})...")
    found_ids, found_dists = index.search(query, k=k, ef_search=50)

    for rank, (doc_id, dist) in enumerate(zip(found_ids[0], found_dists[0]), 1):
        meta = index.get_metadata(doc_id)
        print(f"   Rank {rank}: Label={doc_id:3d}, Dist={dist:.4f}, Category={meta.get('category')}, Price=${meta.get('price'):.2f}")

    # 4. Filtered Vector Search
    filter_expr = {
        "$and": [
            {"category": "books"},
            {"price": {"$lt": 100.0}},
            {"in_stock": True}
        ]
    }
    print(f"\n4. Performing filtered search with AST filter: {filter_expr}...")
    filtered_ids, filtered_dists = index.search(query, k=k, filter=filter_expr, ef_search=50)

    for rank, (doc_id, dist) in enumerate(zip(filtered_ids[0], filtered_dists[0]), 1):
        if doc_id == -1:
            print(f"   Rank {rank}: [No matching neighbor found]")
        else:
            meta = index.get_metadata(doc_id)
            print(f"   Rank {rank}: Label={doc_id:3d}, Dist={dist:.4f}, Category={meta.get('category')}, Price=${meta.get('price'):.2f}, InStock={meta.get('in_stock')}")

    # 5. Persistence (Save & Load)
    save_path = "quickstart_index.hdb"
    print(f"\n5. Serializing index to '{save_path}' with CRC32 integrity checksum...")
    index.save(save_path)
    file_size_kb = os.path.getsize(save_path) / 1024.0
    print(f"   File saved successfully ({file_size_kb:.1f} KB).")

    print(f"   Loading index from '{save_path}'...")
    loaded_index = hnswdb.HNSWIndex.load(save_path)
    assert len(loaded_index) == len(index), "Loaded index size mismatch!"
    print(f"   Loaded index verified: {len(loaded_index)} elements, stats={loaded_index.stats()}")

    # Cleanup demo file
    if os.path.exists(save_path):
        os.remove(save_path)

    print("\n=== Quickstart Completed Successfully ===")


if __name__ == "__main__":
    main()
