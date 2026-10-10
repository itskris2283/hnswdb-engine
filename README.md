# HNSWDB: Approximate Nearest Neighbor Vector Database

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://en.cppreference.com/w/cpp/17)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

**HNSWDB** is an approximate nearest neighbor (ANN) vector database implemented from scratch in C++17 with Python bindings (`pybind11`), a FastAPI REST service, a Typer CLI, and automated test and benchmark suites.

---

## Key Features

- **C++17 Core Engine:** Complete implementation of HNSW graph indexing (Malkov & Yashunin 2018) with heuristic neighbor pruning.
- **SIMD Acceleration:** AVX2 + FMA vector kernels with automatic runtime CPUID dispatch and unrolled scalar fallbacks.
- **Epoch-Based Visited Pool:** Reusable tag pool with $O(1)$ epoch resets, eliminating dynamic heap allocations during graph traversal.
- **Thread-Safe Striped Concurrency:** 1,024 striped spinlocks for deadlock-safe fine-grained edge edits under reader-writer structure locks.
- **Expressive Filtered Search:** JSON/AST filter engine supporting `$eq, $ne, $gt, $gte, $lt, $lte, $in, $and, $or, $not` with inverted index selectivity acceleration.
- **Crash-safe persistence (CRC32 trailer, atomic rename):** Little-endian binary format with IEEE 802.3 CRC32 checksum trailers and atomic write-rename guarantees (`.hdb`).
- **Full-Stack Ecosystem:** Python API with zero-copy NumPy buffers and GIL release, FastAPI collection service, and CLI.

---

## Status and Limitations

- **Storage Architecture:** Memory-resident with binary persistence to `.hdb`; does not support disk-backed mmap or inverted file paging.
- **Capacity Model:** Fixed capacity preallocated on initialization; expandable dynamically via `index.resize()`.
- **Quantization:** Exact 32-bit floating point storage only; no scalar quantization (SQ8) or product quantization (PQ).
- **Execution Target:** CPU-only (x86_64 AVX2/FMA with scalar fallback); no GPU acceleration.
- **Distributed Topology:** Single-node embedded library and server; no distributed partitioning or Raft clustering.
- **Known Graph Construction Weakness:** On synthetic data with tight Gaussian clusters ($\sigma=0.1$), graph construction at $ef_c=100$ exhibits lower search reachability than reference baselines. See [INVESTIGATIONS.md](docs/INVESTIGATIONS.md) for full empirical analysis and isolation experiments.
- **Platform Verification:** Tested on Windows 11 (MSVC 2022). Linux sanitizer CI workflows (ASan, UBSan, TSan) are defined in `.github/workflows/ci.yml`.

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
dim = 16
index = hnswdb.HNSWIndex(dim=dim, metric="l2", max_elements=1000, M=16, ef_construction=100)

# Add vectors with metadata
vectors = np.random.randn(100, dim).astype(np.float32)
ids = np.arange(100, dtype=np.int64)
metadata = [{"category": "books" if i % 2 == 0 else "electronics", "price": i} for i in range(100)]

index.add(vectors, ids, metadata=metadata)

# Search with filter
query = np.random.randn(1, dim).astype(np.float32)
res_ids, res_dists = index.search(query, k=5, filter={"category": "books", "price": {"$lt": 50}})

print("Top neighbors:", res_ids[0])
print("Distances:", res_dists[0])

# Save and load
index.save("my_index.hdb")
loaded_index = hnswdb.HNSWIndex.load("my_index.hdb")
```

### 3. Command Line Interface (CLI)

```bash
# Build index from .npy or .fvecs
hnswdb build --input data.npy --out index.hdb --dim 16 --metric l2 --M 16 --ef-construction 100

# Inspect index statistics
hnswdb info --index index.hdb

# Benchmark search throughput
hnswdb bench --index index.hdb --queries queries.npy --k 10 --ef-search 50

# Start REST server
hnswdb serve --host 0.0.0.0 --port 8000
```

### 4. REST API

<!-- verify:rest -->
```bash
# 1. Create collection
curl -X POST http://localhost:8000/collections/products \
  -H "Content-Type: application/json" \
  -d '{"dim": 4, "metric": "l2", "max_elements": 1000}'

# 2. Insert vectors with metadata
curl -X POST http://localhost:8000/collections/products/insert \
  -H "Content-Type: application/json" \
  -d '{"vectors": [[0.1, 0.2, 0.3, 0.4]], "ids": [101], "metadata": [{"category": "books", "price": 25.0}]}'

# 3. Query collection
curl -X POST http://localhost:8000/collections/products/query \
  -H "Content-Type: application/json" \
  -d '{"vector": [0.1, 0.2, 0.3, 0.4], "k": 5, "filter": {"price": {"$lt": 50}}}'
```

---

## Empirical Benchmark Results

Evaluated on **Fashion-MNIST** (60,000 train vectors, 10,000 test vectors, 784 dimensions, Euclidean L2 distance):

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
ctest --test-dir build/rel -C Release --output-on-failure

# Run Python PyTest suite
pytest tests/python -v
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
