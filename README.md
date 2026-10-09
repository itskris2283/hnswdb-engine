# HNSWDB: High-Performance Approximate Nearest Neighbor Vector Database

[![CI](https://github.com/your-org/hnswdb/actions/workflows/ci.yml/badge.svg)](https://github.com/your-org/hnswdb/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://en.cppreference.com/w/cpp/17)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

**HNSWDB** is an approximate nearest neighbor (ANN) vector database built from scratch in C++17 with Python bindings (pybind11), FastAPI REST service, Typer CLI, and comprehensive test and benchmark suites.

---

## Key Features

- **C++17 Core Engine:** Complete implementation of HNSW (Algorithms 1, 2, 4) with heuristic neighbor pruning and `keepPrunedConnections`.
- **SIMD Acceleration:** Dual AVX2+FMA vector kernels with automatic runtime CPUID dispatch and unrolled scalar fallbacks.
- **Lock-Free Epoch Visited Pool:** $O(1)$ reset tag-pool eliminating heap allocations during graph traversal.
- **Thread-Safe Striped Concurrency:** 1,024 striped spinlocks for deadlock-free fine-grained link edits under reader-writer structure locks.
- **Expressive Filtered Search:** JSON/AST filter engine supporting `$eq, $ne, $gt, $gte, $lt, $lte, $in, $and, $or, $not` with inverted index selectivity acceleration.
- **Production Persistence:** Little-endian binary format with IEEE 802.3 CRC32 checksum trailers and atomic write-rename guarantees.
- **Full-Stack Ecosystem:** Python API with zero-copy NumPy buffers and GIL release, FastAPI collection service, and CLI.

---

## Quickstart

### 1. Installation

```bash
# Clone repository
git clone https://github.com/your-org/hnswdb.git
cd hnswdb

# Install Python package in editable mode
pip install -e .[dev]
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
index.save("my_index.hnsw")
loaded_index = hnswdb.HNSWIndex.load("my_index.hnsw")
```

### 3. Command Line Interface (CLI)

```bash
# Build index from .npy or .fvecs
hnswdb build --input data.npy --out index.hnsw --dim 128 --metric l2 --M 16 --ef-construction 100

# Inspect index statistics
hnswdb info --index index.hnsw

# Benchmark search throughput
hnswdb bench --index index.hnsw --queries queries.npy --k 10 --ef-search 50

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
  -d '{"vector": [...], "k": 10, "filter": {"price": {"$lt": 50}}}'
```

---

## Performance Summary

Evaluated on 20,000 vectors, 64 dimensions, Euclidean L2 distance (AVX2+FMA enabled):

| Metric | HNSWDB | FAISS HNSW | hnswlib |
| :--- | :--- | :--- | :--- |
| **Recall@10 (ef=50)** | **0.989** | 0.997 | 0.996 |
| **p50 Latency** | **0.056 ms** | 0.162 ms | 0.068 ms |
| **Single-Thread QPS** | **20,530** | 18,740 | 22,480 |
| **8-Thread QPS** | **88,410** | 95,120 | 92,110 |
| **Build Time** | **2.56 s** | 0.75 s | 0.41 s |
| **Disk Size** | **7.68 MB** | 7.63 MB | 7.72 MB |

For detailed analysis and trade-offs, see [`docs/TRADEOFFS.md`](docs/TRADEOFFS.md) and [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md).

---

## Testing

```bash
# Run C++ GoogleTest suite
ctest --test-dir build/rel -C Release --output-on-failure

# Run Python PyTest suite
pytest tests/python -v
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
