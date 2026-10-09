#!/usr/bin/env python3
"""
HNSWDB Semantic Search & RAG Demo
Demonstrates document chunk indexing, cosine similarity semantic search,
metadata tagging, and context retrieval.
"""

import numpy as np
import hnswdb


def dummy_text_embedder(texts, dim=64, seed=42):
    """Deterministic pseudo-embeddings for sample documents."""
    rng = np.random.default_rng(seed)
    embeddings = []
    for text in texts:
        # Create deterministic embedding vector based on hash
        h = hash(text) % 10000
        vec = np.sin(np.linspace(h, h + 10, dim)).astype(np.float32)
        norm = np.linalg.norm(vec)
        embeddings.append(vec / (norm + 1e-12))
    return np.array(embeddings, dtype=np.float32)


def main():
    print("=== Semantic Document Search with HNSWDB ===")

    # Sample Knowledge Base Corpus
    documents = [
        {"id": 0, "title": "HNSW Algorithm Overview", "category": "engineering", "text": "Hierarchical Navigable Small World graphs provide logarithmic search complexity."},
        {"id": 1, "title": "SIMD Vector Kernels", "category": "engineering", "text": "AVX2 and FMA intrinsics accelerate Euclidean and Dot Product distance computations."},
        {"id": 2, "title": "Database Crash Safety", "category": "storage", "text": "Atomic rename and IEEE 802.3 CRC32 checksums guarantee zero corruption on unexpected restarts."},
        {"id": 3, "title": "Python Packaging & PyBind11", "category": "tooling", "text": "Scikit-build-core compiles C++17 modules into zero-copy NumPy Python extensions."},
        {"id": 4, "title": "Thread Synchronization", "category": "engineering", "text": "Striped spinlocks provide fine-grained lock striping for concurrent graph edge updates."},
        {"id": 5, "title": "Metadata Filtering AST", "category": "storage", "text": "Inverted index selectivity estimates dynamically switch between graph search and brute-force scans."},
    ]

    dim = 64
    texts = [doc["text"] for doc in documents]
    embeddings = dummy_text_embedder(texts, dim=dim)

    # 1. Initialize Cosine Metric Index
    print(f"\n1. Initializing HNSWDB Cosine Index (dim={dim})...")
    index = hnswdb.HNSWIndex(dim=dim, metric="cosine", max_elements=100, M=16, ef_construction=100)

    # 2. Ingest documents
    labels = np.array([doc["id"] for doc in documents], dtype=np.int64)
    metadata = [{"title": doc["title"], "category": doc["category"], "text": doc["text"]} for doc in documents]

    print(f"2. Indexing {len(documents)} documents into knowledge base...")
    index.add(embeddings, labels, metadata=metadata)

    # 3. Query Semantic Search
    query_text = "How does HNSW perform fast vector search with SIMD?"
    query_vec = dummy_text_embedder([query_text], dim=dim)

    print(f"\n3. Query: '{query_text}'")
    print("   Retrieving top-3 most relevant documents...")
    found_ids, found_dists = index.search(query_vec, k=3, ef_search=50)

    for rank, (doc_id, dist) in enumerate(zip(found_ids[0], found_dists[0]), 1):
        meta = index.get_metadata(doc_id)
        similarity = 1.0 - dist  # Cosine similarity
        print(f"\n   [Result {rank}] (Similarity: {similarity:.4f}, Dist: {dist:.4f})")
        print(f"   Title:    {meta['title']}")
        print(f"   Category: {meta['category']}")
        print(f"   Snippet:  {meta['text']}")

    # 4. Filtered Query
    cat_filter = {"category": "storage"}
    print(f"\n4. Query with category filter: {cat_filter}...")
    filt_ids, filt_dists = index.search(query_vec, k=2, filter=cat_filter)
    for rank, (doc_id, dist) in enumerate(zip(filt_ids[0], filt_dists[0]), 1):
        if doc_id != -1:
            meta = index.get_metadata(doc_id)
            print(f"   [Filtered {rank}] Title: {meta['title']} | Snippet: {meta['text']}")

    print("\n=== Semantic Search Demo Completed ===")


if __name__ == "__main__":
    main()
