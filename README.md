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

<!-- BEGIN GENERATED table=summary_fashion_mnist run_id=fashion_mnist_20261010_174828_0a9f92db -->
**Dataset**: fashion_mnist (60000 x 784d) | **M**=16 | **efC**=100 | **k**=10

| Metric | HNSWDB | FAISS HNSW | hnswlib | Analysis & Notes |
| :--- | :---: | :---: | :---: | :--- |
| **Recall@10 (ef=50)** | 0.9951 | 0.9948 | 0.9951 | Competitive parity |
| **p50 Latency (ef=50)** | 0.220 ms | 0.230 ms | 0.296 ms | Sub-millisecond graph search |
| **Build Time** | 29.94 s | 28.23 s | 28.42 s | Single-thread build |
| **Process RSS Delta** | 194.8 MB | 188.9 MB | 201.9 MB | Memory footprint |
<!-- END GENERATED -->

### Equal-Recall Performance (Targets 0.95 and 0.99)

<!-- BEGIN GENERATED table=readme_equal_recall run_id=equal_recall_20261010_174753 -->
| Dataset | Target Recall | Engine | Status | Interp. p50 (ms) | Interp. Seq QPS | 4T Batch QPS |
| :--- | ---: | :--- | :--- | ---: | ---: | ---: |
| Dataset B (64d, σ=0.1) | 0.95 | **faiss** | achieved | 0.1979 | 2,386 | 64,267 |
| Dataset B (64d, σ=0.1) | 0.99 | **faiss** | achieved | 0.2799 | 2,205 | 44,580 |
| Dataset B (64d, σ=0.1) | 0.95 | **hnswdb** | achieved | 0.0448 | 21,849 | 67,998 |
| Dataset B (64d, σ=0.1) | 0.99 | **hnswdb** | achieved | 0.0640 | 15,291 | 57,963 |
| Dataset B (64d, σ=0.1) | 0.95 | **hnswlib** | achieved | 0.0456 | 20,644 | 53,412 |
| Dataset B (64d, σ=0.1) | 0.99 | **hnswlib** | achieved | 0.0538 | 18,279 | 36,850 |
| Fashion-MNIST (784d) | 0.95 | **faiss** | achieved | 0.1418 | 6,550 | 15,128 |
| Fashion-MNIST (784d) | 0.99 | **faiss** | achieved | 0.2186 | 4,333 | 8,432 |
| Fashion-MNIST (784d) | 0.95 | **hnswdb** | achieved | 0.1319 | 7,073 | 19,681 |
| Fashion-MNIST (784d) | 0.99 | **hnswdb** | achieved | 0.2075 | 4,513 | 10,687 |
| Fashion-MNIST (784d) | 0.95 | **hnswlib** | achieved | 0.1539 | 6,136 | 17,881 |
| Fashion-MNIST (784d) | 0.99 | **hnswlib** | achieved | 0.2750 | 3,365 | 9,654 |
<!-- END GENERATED -->

### Clustered Data Evaluation (5x5 Seed Matrix, Dataset B)

<!-- BEGIN GENERATED table=readme_clustered_matrix run_id=seed-matrix-dbead162 -->
| Engine | 1T Recall@10 (Mean ± Std) | 4T Recall@10 (Mean ± Std) | Unreachable Nodes (Self-Retrieval Mean) |
| :--- | :---: | :---: | :---: |
| **HNSWDB (Cured)** | 0.9998 ± 0.0005 | 0.9985 ± 0.0008 | 9.32 |
| **FAISS HNSW** | 0.9999 ± 0.0002 | 0.9999 ± 0.0002 | 5.40 |
| **hnswlib** | 0.9993 ± 0.0015 | 0.9992 ± 0.0029 | 3.00 |
| HNSWDB (Legacy) | 0.9972 ± 0.0037 | 0.9970 ± 0.0037 | 166.16 |
<!-- END GENERATED -->

### Where HNSWDB Loses Relative to Baselines

<!-- BEGIN GENERATED table=where_it_loses run_id=where_it_loses_c7b365cb -->
| Workload | Metric | HNSWDB | Best baseline (engine) | Relative gap | Source (raw file and key path) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Dataset A (5K x 32d) @ R=0.95 | Sequential QPS (equal recall) | 47374.2 QPS | 50858.0 QPS (hnswlib) | **-6.9%** | `equal_recall_grid.json:interpolations[dataset=A,target_recall=0.95].interpolated_sequential_qps` |
| Dataset A (5K x 32d) @ R=0.95 | 4-thread Batch QPS (equal recall) | 108098.7 QPS | 234608.0 QPS (hnswlib) | **-53.9%** | `equal_recall_grid.json:interpolations[dataset=A,target_recall=0.95].interpolated_batch_qps_4th` |
| Dataset A (5K x 32d) @ R=0.95 | 8-thread Batch QPS (equal recall) | 122032.4 QPS | 269838.3 QPS (faiss) | **-54.8%** | `equal_recall_grid.json:interpolations[dataset=A,target_recall=0.95].interpolated_batch_qps_8th` |
| Dataset A (5K x 32d) @ R=0.99 | 4-thread Batch QPS (equal recall) | 81103.8 QPS | 146686.4 QPS (hnswlib) | **-44.7%** | `equal_recall_grid.json:interpolations[dataset=A,target_recall=0.99].interpolated_batch_qps_4th` |
| Dataset A (5K x 32d) @ R=0.99 | 8-thread Batch QPS (equal recall) | 97362.1 QPS | 152727.9 QPS (hnswlib) | **-36.3%** | `equal_recall_grid.json:interpolations[dataset=A,target_recall=0.99].interpolated_batch_qps_8th` |
| Dataset B (20K x 64d, sigma=0.1) @ R=0.99 | Sequential QPS (equal recall) | 15291.1 QPS | 18279.4 QPS (hnswlib) | **-16.3%** | `equal_recall_grid.json:interpolations[dataset=B_s01,target_recall=0.99].interpolated_sequential_qps` |
| Dataset A (5K x 32d) | Process RSS delta | 1.66 MB | 0.94 MB (FAISS) | **+76.6%** | `bench_A_a9f1a6f6.json:engines[hnswdb].seed_42.rss_delta_mb` |
| Dataset B (20K x 64d, sigma=0.1) | Single-thread build time | 1.4808 s | 1.0645 s (FAISS) | **+39.1%** | `bench_B_s01_8a2302be.json:engines[hnswdb].seed_42.build_time_s` |
| Dataset B (20K x 64d, sigma=0.1) | Process RSS delta | 9.58 MB | 8.11 MB (FAISS) | **+18.1%** | `bench_B_s01_8a2302be.json:engines[hnswdb].seed_42.rss_delta_mb` |
| Dataset B (20K x 64d, sigma=0.1) (ef_search=10) | Recall@10 | 0.8461 | 0.8513 (hnswlib) | **-0.0052 abs** | `bench_B_s01_8a2302be.json:engines[hnswdb].seed_42.ef_results[ef=10].recall_at_k` |
| Fashion-MNIST (60K x 784d) | Single-thread build time | 29.9403 s | 28.2293 s (FAISS) | **+6.1%** | `bench_fashion_mnist_0a9f92db.json:engines[hnswdb].seed_42.build_time_s` |
| Dataset B (20K x 64d, sigma=0.1) | 4-thread build time | 2.522 s | 1.809 s (hnswlib) | **+39.4%** | `multithread_build.json:dataset_b.hnswdb_build_s` |
<!-- END GENERATED -->

For detailed analysis, investigations, and trade-offs, see [TRADEOFFS.md](docs/TRADEOFFS.md), [BENCHMARKS.md](docs/BENCHMARKS.md), and [INVESTIGATIONS.md](docs/INVESTIGATIONS.md). Relative speed varies between runs on this host; treat as parity.

---

## Testing

```bash
# Run C++ GoogleTest suite
ctest --test-dir build/rel -C Release --output-on-failure

# Run Python PyTest suite
python -m pytest tests/python -v

# Primary benchmarks
python scripts/run_pipeline.py --config full --datasets fashion_mnist,B,A --threads 1,2,4,8

# Clustered-data study & ablations
python benchmarks/run_seed_matrix.py
python benchmarks/run_ablations.py
python benchmarks/run_decomposition.py

# Work counting and query ef-semantics
python benchmarks/run_work_counting.py

# Multi-thread build & filtered benchmark
python benchmarks/run_multithread_build.py
python benchmarks/run_filtered_bench.py

# Self-retrieval measurements (engine and baselines)
python scripts/measure_fashion_mnist_self_retrieval.py
python benchmarks/measure_baseline_self_retrieval.py

# Interpolations & stability evaluations
python benchmarks/equal_recall.py
python benchmarks/run_ranking_stability.py
python scripts/where_it_loses.py

# Scorecard & documentation generation
python scripts/make_scorecard.py
python scripts/render_docs.py
python scripts/make_resume_bullets.py

# Complete verification and provenance checkers
python scripts/check_targets_unchanged.py
python scripts/check_claims.py
python scripts/check_provenance.py
python scripts/check_docs_links.py
python scripts/check_repro_commands.py
python scripts/verify_readme.py
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
