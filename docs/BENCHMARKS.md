# HNSWDB Benchmark Report

This document records exact, reproducible benchmark metrics executed on HNSWDB, FAISS HNSW, and hnswlib.

All runs were executed on **Windows 11 x86_64** (Intel Core i7-8650U, MSVC 2022 Community cl.exe 19.40) with **AVX2 + FMA** SIMD acceleration enabled. Raw JSON result artifacts are registered in [MANIFEST.json](../results/raw/MANIFEST.json).

---

## 1. Benchmark Execution Environment & Methodology

- **CPU:** AMD64 Processor (AVX2 + FMA hardware intrinsics)
- **Compilers:** MSVC 2022 Community (cl.exe 19.40), CMake 4.4.4
- **Call-Path Standardization:**
  - Single-query latency: 1-query contiguous array slices executed through native single-query invocation with explicit thread control (`omp_set_num_threads(1)` / `set_num_threads(1)` / `num_threads=1`).
  - Batch QPS: multi-threaded batched queries over 3 repeats with min/median/max reporting.
  - Memory Tracking: Process RSS delta measured via `psutil`.

---

## 2. Benchmark Results

### 2.1 Dataset: `fashion-mnist-784-euclidean`
- **Specification:** 60,000 train vectors, 10,000 test queries, 784 dimensions, Euclidean L2 distance.
- **SHA-256:** `7f24e5122b71346346686a6ea80e08d9145b9c020f9f2db4c4c799dfcfc96a07`

#### Accuracy, Latency & Throughput ($M=16, ef_c=100, k=10$)

| Engine | $ef_{search}$ | Recall@10 | Approx Ratio | p50 Lat (ms) | p99 Lat (ms) | Seq QPS | Batch QPS (1T) | Batch QPS (4T) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **faiss_hnsw** | 10 | 0.9264 | 1.000 | 0.0865 | 0.1643 | 11,086 | 15,755 | 31,664 |
| **faiss_hnsw** | 20 | 0.9761 | 1.000 | 0.1273 | 0.2977 | 6,548 | 10,363 | 20,658 |
| **faiss_hnsw** | 50 | 0.9948 | 1.000 | 0.1948 | 0.4591 | 4,696 | 5,573 | 11,249 |
| **faiss_hnsw** | 100 | 0.9980 | 1.000 | 0.3231 | 0.7120 | 2,889 | 2,575 | 6,741 |
| **faiss_hnsw** | 200 | 0.9990 | 1.000 | 0.5572 | 1.0906 | 1,746 | 1,901 | 3,921 |
| **hnswdb** | 10 | 0.9698 | 1.000 | 0.1320 | 0.4035 | 6,549 | 9,886 | 13,066 |
| **hnswdb** | 20 | 0.9894 | 1.000 | 0.1317 | 0.2628 | 7,299 | 6,816 | 10,946 |
| **hnswdb** | 50 | 0.9972 | 1.000 | 0.2200 | 0.4415 | 4,304 | 3,864 | 9,102 |
| **hnswdb** | 100 | 0.9987 | 1.000 | 0.3738 | 0.7031 | 2,568 | 2,988 | 6,119 |
| **hnswdb** | 200 | 0.9991 | 1.000 | 0.5342 | 1.0114 | 1,792 | 1,803 | 3,901 |
| **hnswlib** | 10 | 0.9263 | 1.000 | 0.1123 | 0.2383 | 8,372 | 10,217 | 27,893 |
| **hnswlib** | 20 | 0.9753 | 1.000 | 0.1483 | 0.2365 | 6,585 | 7,525 | 19,361 |
| **hnswlib** | 50 | 0.9950 | 1.000 | 0.2512 | 0.4161 | 3,899 | 4,334 | 10,044 |
| **hnswlib** | 100 | 0.9982 | 1.000 | 0.3559 | 0.6337 | 2,712 | 2,929 | 6,529 |
| **hnswlib** | 200 | 0.9992 | 1.000 | 0.6659 | 1.7252 | 1,283 | 1,624 | 4,057 |

#### Build Time, Process RSS & Disk Size (Fashion-MNIST 60k)

| Engine | Build Time (s) | Serialized Disk Size (MB) | Process RSS Delta (MB) |
| :--- | :---: | :---: | :---: |
| **faiss_hnsw** | 9.66 s | 187.70 MB | 189.11 MB |
| **hnswdb** | 20.02 s | 187.85 MB | 194.09 MB |
| **hnswlib** | 8.51 s | 187.94 MB | 203.17 MB |

---

### 2.2 Dataset: `synthetic_20000_64_l2` (Dataset B)
- **Specification:** 20,000 train vectors, 1,000 test queries, 64 dimensions, 50 Gaussian clusters, Euclidean L2 distance.

#### Accuracy, Latency & Throughput ($M=16, ef_c=100, k=10$)

| Engine | $ef_{search}$ | Recall@10 | Approx Ratio | p50 Lat (ms) | p99 Lat (ms) | Seq QPS | Batch QPS (1T) | Batch QPS (4T) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **faiss_hnsw** | 10 | 0.8508 | 1.000 | 0.0733 | 0.1160 | 13,239 | 66,896 | 238,720 |
| **faiss_hnsw** | 50 | 0.9943 | 1.000 | 0.0817 | 0.2055 | 9,587 | 19,397 | 61,815 |
| **faiss_hnsw** | 200 | 0.9994 | 1.000 | 0.2074 | 0.5157 | 3,791 | 5,562 | 14,346 |
| **hnswdb** | 10 | 0.9102 | 1.000 | 0.0274 | 0.0896 | 30,871 | 36,186 | 95,090 |
| **hnswdb** | 50 | 0.9839 | 1.000 | 0.0744 | 0.2034 | 11,891 | 23,008 | 76,682 |
| **hnswdb** | 200 | 0.9844 | 1.000 | 0.1129 | 0.2520 | 8,094 | 9,130 | 28,511 |
| **hnswlib** | 10 | 0.8508 | 1.000 | 0.0651 | 0.1966 | 13,555 | 18,965 | 60,979 |
| **hnswlib** | 50 | 0.9968 | 1.000 | 0.1309 | 0.5306 | 6,765 | 7,599 | 27,896 |
| **hnswlib** | 200 | 1.0000 | 1.000 | 0.1230 | 0.2521 | 7,776 | 8,904 | 25,791 |

*(Note: On clustered synthetic distributions with $ef_c=100$, HNSWDB reaches 0.9844 recall floor; with $ef_c=400$, recall reaches 0.9989 as documented in [KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md)).*
