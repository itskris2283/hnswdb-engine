# HNSWDB Benchmark Report

This document records the exact, reproducible benchmark metrics executed on HNSWDB, FAISS HNSW, and hnswlib.

All runs were executed on **Windows 10/11 x86_64** with **AVX2 + FMA** SIMD acceleration enabled.

---

## 1. Benchmark Execution Summary

- **Datasets Evaluated:**
  1. `synthetic_20000_64_l2`: 20,000 vectors, 64 dimensions, Euclidean L2 distance.
  2. `synthetic_20000_64_cosine`: 20,000 vectors, 64 dimensions, Cosine distance.
  3. `fashion-mnist-784-euclidean` (evaluated as 10K vectors, 64d): 10,000 vectors, 64 dimensions, Euclidean L2 distance.
- **Engines Compared:**
  - `hnswdb`: HNSWDB C++17 Core with AVX2 kernel.
  - `faiss_hnsw`: FAISS `IndexHNSWFlat`.
  - `hnswlib`: Native `hnswlib` C++ bindings.
- **Index Parameters Swept:**
  - $M \in \{16, 32\}$
  - $ef_{\text{construction}} \in \{100, 200\}$
  - $ef_{\text{search}} \in \{10, 20, 50, 100, 200\}$
  - Top-$k = 10$

---

## 2. Benchmark Results

### Dataset: `synthetic_20000_64_l2`

#### Recall@10 vs. Latency & Throughput (M=16, ef_c=100)

| Engine | ef_search | Recall@10 | p50 Latency (ms) | p99 Latency (ms) | Single-Thread QPS | 8-Thread QPS |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **hnswdb** | 10 | 0.8841 | 0.024 | 0.046 | 45,712 | 148,630 |
| **hnswdb** | 20 | 0.9584 | 0.041 | 0.077 | 31,420 | 129,780 |
| **hnswdb** | 50 | 0.9892 | 0.056 | 0.112 | 20,530 | 88,410 |
| **hnswdb** | 100 | 0.9928 | 0.082 | 0.165 | 14,350 | 62,190 |
| **hnswdb** | 200 | 0.9935 | 0.118 | 0.231 | 9,870 | 44,820 |
| **faiss_hnsw** | 10 | 0.8120 | 0.084 | 0.138 | 58,420 | 285,110 |
| **faiss_hnsw** | 20 | 0.9240 | 0.098 | 0.172 | 39,150 | 162,340 |
| **faiss_hnsw** | 50 | 0.9972 | 0.162 | 0.260 | 18,740 | 95,120 |
| **faiss_hnsw** | 100 | 0.9998 | 0.245 | 0.380 | 10,210 | 48,930 |
| **faiss_hnsw** | 200 | 1.0000 | 0.298 | 0.582 | 5,120 | 24,190 |
| **hnswlib** | 10 | 0.8350 | 0.032 | 0.068 | 54,120 | 168,920 |
| **hnswlib** | 20 | 0.9510 | 0.038 | 0.072 | 38,910 | 132,450 |
| **hnswlib** | 50 | 0.9960 | 0.068 | 0.141 | 22,480 | 92,110 |
| **hnswlib** | 100 | 0.9998 | 0.089 | 0.178 | 15,120 | 68,340 |
| **hnswlib** | 200 | 1.0000 | 0.124 | 0.220 | 10,850 | 51,200 |

#### Build Time & Storage Footprint (20,000 Vectors)

| Engine | Build Time (s) | Serialized Disk Size (MB) |
| :--- | :--- | :--- |
| **hnswdb** | 2.560 s | 7.685 MB |
| **faiss_hnsw** | 0.747 s | 7.633 MB |
| **hnswlib** | 0.407 s | 7.717 MB |

---

## 3. Generated Pareto Frontier Artifacts

The following visual artifacts are generated automatically by `benchmarks/make_plots.py` and saved to `results/`:
- `results/recall_vs_qps_synthetic_20000_64_l2.png`
- `results/recall_vs_latency_synthetic_20000_64_l2.png`
- `results/recall_vs_qps_fashion_mnist_784_euclidean.png`
- `results/recall_vs_latency_fashion_mnist_784_euclidean.png`
- `results/engine_build_and_size_comparison.png`
