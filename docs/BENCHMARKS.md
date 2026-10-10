# HNSWDB Benchmark Report

This document records exact, reproducible benchmark metrics executed on HNSWDB, FAISS HNSW, and hnswlib.

All runs were executed on **Windows 11 x86_64** (Intel Core i7-8650U, MSVC 2022 Community) with **AVX2 + FMA** SIMD acceleration enabled. Every table in this document is programmatically rendered from verified raw JSON result artifacts registered in [MANIFEST.json](../results/raw/MANIFEST.json). No numbers are hand-typed.

---

## 1. Benchmark Execution Environment & Methodology

- **CPU:** AMD64 Family 6 Model 142 (AVX2 + FMA hardware intrinsics)
- **Compilers:** MSVC 2022 Community (cl.exe 19.40), CMake 4.4.4, Python 3.12
- **Call-Path Standardization:**
  - Single-query latency: 1-query contiguous array slices executed through native single-query invocation with explicit thread control (`num_threads=1` / `omp_set_num_threads(1)` / `set_num_threads(1)`).
  - Batch QPS: multi-threaded batched queries with min/median/max reporting.
  - Memory Tracking: Process RSS delta measured via `psutil`.

---

## 2. Benchmark Results

### 2.1 Dataset: `fashion-mnist-784-euclidean`
- **Specification:** 60,000 train vectors, 10,000 test queries, 784 dimensions, Euclidean L2 distance.
- **Parameters:** $M=16, ef_c=100, k=10$

<!-- BEGIN GENERATED table=full_fashion_mnist run_id=fashion_mnist_20261010_113121_90c0d1d3 -->
**Dataset**: fashion_mnist (60000 x 784d) | **M**=16 | **efC**=100 | **k**=10

| Engine | ef_search | Recall@10 | Approx Ratio | p50 (ms) | Sequential QPS |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **hnswdb** | 10 | 0.9705 | 1.0009 | 0.204 | 4,615 |
| **hnswdb** | 50 | 0.9978 | 1.0001 | 0.452 | 2,085 |
| **hnswdb** | 200 | 0.9996 | 1.0000 | 1.107 | 858 |
| **faiss** | 10 | 0.9264 | 1.0009 | 0.149 | 6,235 |
| **faiss** | 50 | 0.9948 | 1.0001 | 0.365 | 2,588 |
| **faiss** | 200 | 0.9990 | 1.0000 | 1.623 | 492 |
| **hnswlib** | 10 | 0.9277 | 1.0009 | 0.291 | 2,816 |
| **hnswlib** | 50 | 0.9951 | 1.0001 | 0.813 | 1,033 |
| **hnswlib** | 200 | 0.9991 | 1.0000 | 1.941 | 451 |
<!-- END GENERATED -->

#### Build Performance & Resource Utilization

<!-- BEGIN GENERATED table=build_fashion_mnist run_id=fashion_mnist_20261010_113121_90c0d1d3 -->
**Dataset**: fashion_mnist (60000 x 784d) | **M**=16 | **efC**=100 | **k**=10

| Engine | Build Time (s) | RSS Delta (MB) |
| :--- | ---: | ---: |
| hnswdb | 102.599 | 195.1 |
| faiss | 38.206 | 189.2 |
| hnswlib | 78.722 | 202.1 |
<!-- END GENERATED -->

---

### 2.2 Dataset: `Dataset B (sigma=0.1)` (Clustered Synthetic)
- **Specification:** 20,000 train vectors, 1,000 test queries, 64 dimensions, 50 Gaussian clusters ($\sigma=0.1$).
- **Parameters:** $M=16, ef_c=100, k=10$

<!-- BEGIN GENERATED table=full_dataset_b run_id=B_s01_20261010_100918_0d563c34 -->
**Dataset**: B_s01 (20000 x 64d) | **M**=16 | **efC**=100 | **k**=10

| Engine | ef_search | Recall@10 | Approx Ratio | p50 (ms) | Sequential QPS |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **hnswdb** | 10 | 0.9101 | 1.0735 | 0.021 | 43,684 |
| **hnswdb** | 50 | 0.9711 | 1.0017 | 0.039 | 23,262 |
| **hnswdb** | 200 | 0.9716 | 1.0017 | 0.085 | 11,346 |
| **faiss** | 10 | 0.8508 | 1.1503 | 0.035 | 24,424 |
| **faiss** | 50 | 0.9943 | 1.0336 | 0.055 | 17,057 |
| **faiss** | 200 | 0.9994 | 1.0000 | 0.159 | 5,911 |
| **hnswlib** | 10 | 0.8513 | 1.0253 | 0.028 | 31,381 |
| **hnswlib** | 50 | 0.9979 | 1.0001 | 0.057 | 15,685 |
| **hnswlib** | 200 | 1.0000 | 1.0000 | 0.089 | 10,755 |
<!-- END GENERATED -->

#### Build Performance & Resource Utilization

<!-- BEGIN GENERATED table=build_dataset_b run_id=B_s01_20261010_100918_0d563c34 -->
**Dataset**: B_s01 (20000 x 64d) | **M**=16 | **efC**=100 | **k**=10

| Engine | Build Time (s) | RSS Delta (MB) |
| :--- | ---: | ---: |
| hnswdb | 3.173 | 9.9 |
| faiss | 1.303 | 8.3 |
| hnswlib | 1.260 | 15.4 |
<!-- END GENERATED -->

*(Note: On tight clusters with $\sigma=0.1$ and $ef_c=100$, HNSWDB exhibits a recall ceiling around ~0.97 due to heuristic pruning during overflow reselection; at $ef_c=400$, recall converges to 1.0000 as analyzed in [INVESTIGATIONS.md](INVESTIGATIONS.md) and [KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md)).*

---

### 2.3 Dataset: `Dataset A` (Synthetic Benchmark)
- **Specification:** 5,000 train vectors, 1,000 test queries, 32 dimensions, 20 Gaussian clusters ($\sigma=0.1$).
- **Parameters:** $M=16, ef_c=100, k=10$

<!-- BEGIN GENERATED table=full_dataset_a run_id=A_20261010_100807_7d8938d0 -->
**Dataset**: A (5000 x 32d) | **M**=16 | **efC**=100 | **k**=10

| Engine | ef_search | Recall@10 | Approx Ratio | p50 (ms) | Sequential QPS |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **hnswdb** | 10 | 0.9835 | 1.0082 | 0.014 | 66,247 |
| **hnswdb** | 50 | 0.9987 | 1.0000 | 0.026 | 36,574 |
| **hnswdb** | 200 | 0.9987 | 1.0000 | 0.046 | 20,252 |
| **faiss** | 10 | 0.9224 | 1.0118 | 0.019 | 48,528 |
| **faiss** | 50 | 0.9986 | 1.0001 | 0.038 | 22,853 |
| **faiss** | 200 | 0.9990 | 1.0000 | 0.128 | 7,418 |
| **hnswlib** | 10 | 0.9272 | 1.0031 | 0.015 | 65,920 |
| **hnswlib** | 50 | 0.9996 | 1.0000 | 0.026 | 36,405 |
| **hnswlib** | 200 | 1.0000 | 1.0000 | 0.050 | 19,304 |
<!-- END GENERATED -->

#### Build Performance & Resource Utilization

<!-- BEGIN GENERATED table=build_dataset_a run_id=A_20261010_100807_7d8938d0 -->
**Dataset**: A (5000 x 32d) | **M**=16 | **efC**=100 | **k**=10

| Engine | Build Time (s) | RSS Delta (MB) |
| :--- | ---: | ---: |
| hnswdb | 0.584 | 2.0 |
| faiss | 0.185 | 1.6 |
| hnswlib | 0.193 | 7.2 |
<!-- END GENERATED -->
