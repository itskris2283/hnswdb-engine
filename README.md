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
- **Thread-Safe Striped Concurrency:** 1,024 striped spinlocks for deadlock-free fine-grained edge edits under reader-writer structure locks.
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

| Metric | HNSWDB | FAISS HNSW | hnswlib |
| :--- | :---: | :---: | :---: |
| **Recall@10 (ef=50)** | 0.9976 | 0.9948 | 0.9950 |
| **Recall@10 (ef=200)** | 0.9994 | 0.9990 | 0.9992 |
| **p50 Latency (ef=50)** | 0.220 ms | 0.195 ms | 0.251 ms |
| **p99 Latency (ef=50)** | 0.442 ms | 0.459 ms | 0.416 ms |
| **Sequential QPS (ef=50)** | 4,304 | 4,696 | 3,899 |
| **4-Thread Batch QPS (ef=50)** | 9,102 | 11,249 | 10,044 |
| **Build Time (60k items)** | 20.02 s | 9.66 s | 8.51 s |
| **Index Disk Size** | 187.85 MB | 187.70 MB | 187.94 MB |
| **Process RSS Delta** | 194.09 MB | 189.11 MB | 203.17 MB |

For detailed analysis, investigations, and trade-offs, see [`docs/TRADEOFFS.md`](docs/TRADEOFFS.md), [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md), and [`docs/INVESTIGATIONS.md`](docs/INVESTIGATIONS.md).

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
