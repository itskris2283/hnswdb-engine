# HNSWDB: Approximate Nearest Neighbor Vector Database

[![CI](https://github.com/itskris2283/hnswdb/actions/workflows/ci.yml/badge.svg)](https://github.com/itskris2283/hnswdb/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://en.cppreference.com/w/cpp/17)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

**HNSWDB** is an approximate nearest neighbor (ANN) vector database implemented from scratch in C++17 with Python bindings (`pybind11`), a FastAPI REST service, a Typer CLI, and rigorous automated test and benchmark suites.

---

## Key Features

- **C++17 Core Engine:** Complete implementation of HNSW graph indexing (Malkov & Yashunin 2018) with heuristic neighbor pruning and `keepPrunedConnections`.
- **SIMD Acceleration:** AVX2 + FMA vector kernels with automatic runtime CPUID dispatch and unrolled scalar fallbacks.
- **Epoch-Based Visited Pool:** Reusable tag-pool with $O(1)$ epoch resets, eliminating dynamic heap allocations during graph traversal.
- **Thread-Safe Striped Concurrency:** 1,024 striped spinlocks for deadlock-safe fine-grained edge edits under reader-writer structure locks.

- **Expressive Filtered Search:** JSON/AST filter engine supporting `$eq, $ne, $gt, $gte, $lt, $lte, $in, $and, $or, $not` with inverted index selectivity acceleration.
- **Production Persistence:** Little-endian binary format with IEEE 802.3 CRC32 checksum trailers and atomic write-rename guarantees (`.hdb`).
- **Full-Stack Ecosystem:** Python API with zero-copy NumPy buffers and GIL release, FastAPI collection service, and CLI.

---

## Quickstart

### 1. Installation

```bash
# Clone repository
git clone https://github.com/itskris2283/hnswdb.git
cd hnswdb

# Install Python package in editable mode
pip install -e ".[dev]"
```

### 2. Python API

```python
import numpy as np
import hnswdb

# Create HNSW index
dim = 128
index = hnswdb.HNSWIndex(dim=dim, metric="l2", max_elements=10000, M=16, ef_construction=100)

# Add vectors with metadata
vectors = np.random.randn(1000, dim).astype(np.float32)
ids = np.arange(1000, dtype=np.int64)
metadata = [{"category": "books" if i % 2 == 0 else "electronics", "price": i} for i in range(1000)]

index.add(vectors, ids, metadata=metadata)

# Search with filter
query = np.random.randn(1, dim).astype(np.float32)
res_ids, res_dists = index.search(query, k=5, filter={"category": "books", "price": {"$lt": 200}})

print("Top neighbors:", res_ids[0])
print("Distances:", res_dists[0])

# Save and load
index.save("my_index.hdb")
loaded_index = hnswdb.HNSWIndex.load("my_index.hdb")
```

### 3. Command Line Interface (CLI)

```bash
# Build index from .npy or .fvecs
hnswdb build --input data.npy --out index.hdb --dim 128 --metric l2 --M 16 --ef-construction 100

# Inspect index statistics
hnswdb info --index index.hdb

# Benchmark search throughput
hnswdb bench --index index.hdb --queries queries.npy --k 10 --ef-search 50

# Start REST server
hnswdb serve --host 0.0.0.0 --port 8000
```

### 4. REST API

```bash
# Create collection
curl -X POST http://localhost:8000/collections/products \
  -H "Content-Type: application/json" \
  -d '{"dim": 128, "metric": "l2", "max_elements": 10000}'

# Query collection
curl -X POST http://localhost:8000/collections/products/query \
  -H "Content-Type: application/json" \
  -d '{"vector": [0.1, 0.2, ...], "k": 10, "filter": {"price": {"$lt": 50}}}'
```

---

## Empirical Benchmark Results

Evaluated on **Fashion-MNIST** (60,000 train vectors, 10,000 test vectors, 784 dimensions, Euclidean L2 distance; SHA-256 `7f24e512...`):

<!-- BEGIN GENERATED table=summary_fashion_mnist run_id=fashion_mnist_20261010_113121_90c0d1d3 -->
**Dataset**: fashion_mnist (60000 x 784d) | **M**=16 | **efC**=100 | **k**=10

| Metric | HNSWDB | FAISS HNSW | hnswlib | Analysis & Notes |
| :--- | :---: | :---: | :---: | :--- |
| **Recall@10 (ef=50)** | 0.9978 | 0.9948 | 0.9951 | Competitive parity |
| **p50 Latency (ef=50)** | 0.452 ms | 0.365 ms | 0.813 ms | Sub-millisecond graph search |
| **Build Time** | 102.60 s | 38.21 s | 78.72 s | Single-thread build |
| **Process RSS Delta** | 195.1 MB | 189.2 MB | 202.1 MB | Memory footprint |
<!-- END GENERATED -->

For detailed analysis, investigations, and trade-offs, see [TRADEOFFS.md](docs/TRADEOFFS.md), [BENCHMARKS.md](docs/BENCHMARKS.md), and [INVESTIGATIONS.md](docs/INVESTIGATIONS.md).


---

## Testing

```bash
# Run C++ GoogleTest suite
ctest --test-dir build -C Release --output-on-failure

# Run Python PyTest suite
pytest tests/python -v
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
