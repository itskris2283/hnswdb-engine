# HNSWDB Benchmark Report

This document records exact, reproducible benchmark metrics executed on HNSWDB, FAISS HNSW, and hnswlib.

All runs were executed on **Windows 11 x86_64** (Intel Core i7-8650U, MSVC 2022 Community) with **AVX2 + FMA** SIMD acceleration enabled. Every table in this document is programmatically rendered from verified raw JSON result artifacts registered in [MANIFEST.json](../results/raw/MANIFEST.json). No numbers are hand-typed.

---

## 1. Benchmark Execution Environment & Methodology

<!-- BEGIN GENERATED table=hardware_environment run_id=fashion_mnist_20261010_174828_0a9f92db -->
**Dataset**: fashion_mnist (60000 x 784d) | **M**=16 | **efC**=100 | **k**=10

| Component | Specification |
| :--- | :--- |
| **CPU Model** | Intel64 Family 6 Model 142 Stepping 10, GenuineIntel |
| **Core Topology** | 4 physical cores, 8 logical cores |
| **Base Clock** | 1910.0 MHz (nominal 1.90 GHz, boost up to 4.20 GHz) |
| **Platform / OS** | Windows-11-10.0.22621-SP0 |
| **Compiler** | MSVC 2022 (cl.exe 19.40, /O2 /MD /arch:AVX2) |
| **Active SIMD Kernel** | AVX2 + FMA (CPUID runtime dispatch) |
| **Throttling Detected** | False (pinned high-performance profile) |
<!-- END GENERATED -->

- **Call-Path Standardization:**
  - Single-query latency: 1-query contiguous array slices executed through native single-query invocation with explicit thread control (`num_threads=1` / `omp_set_num_threads(1)` / `set_num_threads(1)`).
  - Batch QPS: multi-threaded batched queries with min/median/max reporting across 1, 2, 4 physical threads and 8 logical (HT).
  - Memory Tracking: Process RSS delta measured via `psutil`.

### 1.1 ef Semantics: Distance Computations per Query (HNSWDB vs FAISS)

Verifies whether identical $ef_{search}$ parameter implies equal algorithmic distance work:

<!-- BEGIN GENERATED table=ef_semantics run_id=query_ef_semantics_20261010_181658_2464a233 -->
**Dataset**: unknown (? x ?d) | **M**=? | **efC**=? | **k**=?

| Dataset | ef_search | HNSWDB Distances / Query | FAISS Distances / Query | Ratio (HNSWDB / FAISS) | Work Invariance Status |
| :--- | ---: | ---: | ---: | ---: | :--- |
| A | 10 | 336.3 | 340.8 | 0.99x | Equivalent (+/- 5%) |
| A | 50 | 949.8 | 977.9 | 0.97x | Equivalent (+/- 5%) |
| A | 100 | 1533.4 | 1567.7 | 0.98x | Equivalent (+/- 5%) |
| A | 200 | 2351.2 | 2380.2 | 0.99x | Equivalent (+/- 5%) |
| B_s01 | 10 | 437.9 | 407.8 | 1.07x | Equivalent (+/- 10%) |
| B_s01 | 50 | 1322.9 | 1255.4 | 1.05x | Equivalent (+/- 10%) |
| B_s01 | 100 | 2260.6 | 2154.9 | 1.05x | Equivalent (+/- 10%) |
| B_s01 | 200 | 3819.7 | 3647.0 | 1.05x | Equivalent (+/- 10%) |
| fashion_mnist | 10 | 222.8 | 212.6 | 1.05x | Equivalent (+/- 10%) |
| fashion_mnist | 50 | 501.8 | 491.4 | 1.02x | Equivalent (+/- 5%) |
| fashion_mnist | 100 | 763.7 | 751.4 | 1.02x | Equivalent (+/- 5%) |
| fashion_mnist | 200 | 1175.0 | 1159.5 | 1.01x | Equivalent (+/- 5%) |
<!-- END GENERATED -->

---

## 2. Benchmark Results

### 2.1 Dataset: `fashion-mnist-784-euclidean`
- **Specification:** 60,000 train vectors, 10,000 test queries, 784 dimensions, Euclidean L2 distance.
- **Parameters:** $M=16, ef_c=100, k=10$

<!-- BEGIN GENERATED table=full_fashion_mnist run_id=fashion_mnist_20261010_174828_0a9f92db -->
**Dataset**: fashion_mnist (60000 x 784d) | **M**=16 | **efC**=100 | **k**=10

| Engine | ef_search | Recall@10 | Approx Ratio | p50 (ms) | Sequential QPS |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **hnswdb** | 10 | 0.9278 | 1.0034 | 0.103 | 9,071 |
| **hnswdb** | 50 | 0.9951 | 1.0003 | 0.220 | 4,259 |
| **hnswdb** | 200 | 0.9992 | 1.0000 | 0.478 | 2,048 |
| **faiss** | 10 | 0.9264 | 1.0032 | 0.110 | 8,357 |
| **faiss** | 50 | 0.9948 | 1.0003 | 0.230 | 4,125 |
| **faiss** | 200 | 0.9990 | 1.0001 | 0.614 | 1,519 |
| **hnswlib** | 10 | 0.9277 | 1.0030 | 0.111 | 8,577 |
| **hnswlib** | 50 | 0.9951 | 1.0002 | 0.296 | 3,118 |
| **hnswlib** | 200 | 0.9991 | 1.0000 | 0.706 | 1,346 |
<!-- END GENERATED -->

#### Build Performance & Resource Utilization

<!-- BEGIN GENERATED table=build_fashion_mnist run_id=fashion_mnist_20261010_174828_0a9f92db -->
**Dataset**: fashion_mnist (60000 x 784d) | **M**=16 | **efC**=100 | **k**=10

| Engine | Build Time (s) | RSS Delta (MB) |
| :--- | ---: | ---: |
| hnswdb | 29.940 | 194.8 |
| faiss | 28.229 | 188.9 |
| hnswlib | 28.420 | 201.9 |
<!-- END GENERATED -->

---

### 2.2 Dataset: `Dataset B (sigma=0.1)` (Clustered Synthetic)
- **Specification:** 20,000 train vectors, 1,000 test queries, 64 dimensions, 50 Gaussian clusters ($\sigma=0.1$).
- **Parameters:** $M=16, ef_c=100, k=10$

<!-- BEGIN GENERATED table=full_dataset_b run_id=B_s01_20261010_174800_8a2302be -->
**Dataset**: B_s01 (20000 x 64d) | **M**=16 | **efC**=100 | **k**=10

| Engine | ef_search | Recall@10 | Approx Ratio | p50 (ms) | Sequential QPS |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **hnswdb** | 10 | 0.8461 | 1.0539 | 0.018 | 55,211 |
| **hnswdb** | 50 | 0.9978 | 1.0001 | 0.069 | 14,263 |
| **hnswdb** | 200 | 1.0000 | 1.0000 | 0.090 | 11,029 |
| **faiss** | 10 | 0.8508 | 1.1503 | 0.084 | 2,905 |
| **faiss** | 50 | 0.9943 | 1.0336 | 0.290 | 2,186 |
| **faiss** | 200 | 0.9994 | 1.0000 | 0.387 | 1,571 |
| **hnswlib** | 10 | 0.8513 | 1.0253 | 0.030 | 27,869 |
| **hnswlib** | 50 | 0.9979 | 1.0001 | 0.056 | 17,846 |
| **hnswlib** | 200 | 1.0000 | 1.0000 | 0.106 | 9,258 |
<!-- END GENERATED -->

#### Build Performance & Resource Utilization

<!-- BEGIN GENERATED table=build_dataset_b run_id=B_s01_20261010_174800_8a2302be -->
**Dataset**: B_s01 (20000 x 64d) | **M**=16 | **efC**=100 | **k**=10

| Engine | Build Time (s) | RSS Delta (MB) |
| :--- | ---: | ---: |
| hnswdb | 1.481 | 9.6 |
| faiss | 1.065 | 8.1 |
| hnswlib | 3.382 | 15.3 |
<!-- END GENERATED -->

---

### 2.3 Dataset: `Dataset A` (Synthetic Benchmark)
- **Specification:** 5,000 train vectors, 1,000 test queries, 32 dimensions, 20 Gaussian clusters ($\sigma=0.1$).
- **Parameters:** $M=16, ef_c=100, k=10$

<!-- BEGIN GENERATED table=full_dataset_a run_id=A_20261010_174754_a9f1a6f6 -->
**Dataset**: A (5000 x 32d) | **M**=16 | **efC**=100 | **k**=10

| Engine | ef_search | Recall@10 | Approx Ratio | p50 (ms) | Sequential QPS |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **hnswdb** | 10 | 0.9267 | 1.0031 | 0.017 | 57,048 |
| **hnswdb** | 50 | 0.9995 | 1.0000 | 0.028 | 31,923 |
| **hnswdb** | 200 | 1.0000 | 1.0000 | 0.055 | 13,757 |
| **faiss** | 10 | 0.9224 | 1.0118 | 0.023 | 42,079 |
| **faiss** | 50 | 0.9986 | 1.0001 | 0.041 | 23,729 |
| **faiss** | 200 | 0.9990 | 1.0000 | 0.138 | 6,666 |
| **hnswlib** | 10 | 0.9272 | 1.0031 | 0.015 | 63,763 |
| **hnswlib** | 50 | 0.9996 | 1.0000 | 0.030 | 31,097 |
| **hnswlib** | 200 | 1.0000 | 1.0000 | 0.057 | 16,975 |
<!-- END GENERATED -->

#### Build Performance & Resource Utilization

<!-- BEGIN GENERATED table=build_dataset_a run_id=A_20261010_174754_a9f1a6f6 -->
**Dataset**: A (5000 x 32d) | **M**=16 | **efC**=100 | **k**=10

| Engine | Build Time (s) | RSS Delta (MB) |
| :--- | ---: | ---: |
| hnswdb | 0.165 | 1.7 |
| faiss | 0.234 | 0.9 |
| hnswlib | 0.249 | 7.2 |
<!-- END GENERATED -->

---

### 2.4 Multi-Threaded Build Performance (Target T9)

Evaluated at 4 physical threads (host maximum) with median of 3 repeats:

<!-- BEGIN GENERATED table=multithread_build run_id=multithread_build_20261010_180617_73c07096 -->
**Dataset**: unknown (? x ?d) | **M**=? | **efC**=? | **k**=?

| Dataset | HNSWDB 4T Build (s) | hnswlib 4T Build (s) | Ratio vs hnswlib | Target Status |
| :--- | ---: | ---: | ---: | :--- |
| Dataset B (20K x 64d) | 2.522 | 1.809 | 1.39x | **PASS** |
| Fashion-MNIST (60K x 784d) | 9.162 | 10.456 | 0.88x | **PASS** |
<!-- END GENERATED -->

---

### 2.5 Equal-Recall Comparison (Primary Benchmark)

Log-linear interpolation across $ef_{search} \in \{10, 20, 50, 100, 200, 400\}$ to determine exact latency and throughput at target recall milestones:

<!-- BEGIN GENERATED table=equal_recall run_id=equal_recall_20261010_174753 -->
**Dataset**: unknown (? x ?d) | **M**=? | **efC**=? | **k**=?

| Dataset | Target Recall | Engine | Status | Interp. p50 (ms) | Interp. Seq QPS | 4T Batch QPS | 8T (HT) Batch QPS |
| :--- | ---: | :--- | :--- | ---: | ---: | ---: | ---: |
| A | 0.90 | **faiss** | achieved | 0.0231 | 42,079 | 246,719 | 403,437 |
| A | 0.95 | **faiss** | achieved | 0.0284 | 34,194 | 170,010 | 269,838 |
| A | 0.98 | **faiss** | achieved | 0.0356 | 27,290 | 113,419 | 174,278 |
| A | 0.99 | **faiss** | achieved | 0.0383 | 25,314 | 99,104 | 150,645 |
| A | 0.90 | **hnswdb** | achieved | 0.0165 | 57,048 | 127,792 | 139,190 |
| A | 0.95 | **hnswdb** | achieved | 0.0196 | 47,374 | 108,099 | 122,032 |
| A | 0.98 | **hnswdb** | achieved | 0.0245 | 37,294 | 87,144 | 103,018 |
| A | 0.99 | **hnswdb** | achieved | 0.0264 | 34,436 | 81,104 | 97,362 |
| A | 0.90 | **hnswlib** | achieved | 0.0151 | 63,763 | 306,617 | 283,624 |
| A | 0.95 | **hnswlib** | achieved | 0.0187 | 50,858 | 234,608 | 226,540 |
| A | 0.98 | **hnswlib** | achieved | 0.0249 | 37,769 | 164,960 | 168,548 |
| A | 0.99 | **hnswlib** | achieved | 0.0274 | 34,203 | 146,686 | 152,728 |
| B_s01 | 0.90 | **faiss** | achieved | 0.1283 | 2,635 | 101,520 | 112,366 |
| B_s01 | 0.95 | **faiss** | achieved | 0.1979 | 2,386 | 64,267 | 79,665 |
| B_s01 | 0.98 | **faiss** | achieved | 0.2567 | 2,249 | 48,849 | 64,811 |
| B_s01 | 0.99 | **faiss** | achieved | 0.2799 | 2,205 | 44,580 | 60,503 |
| B_s01 | 0.90 | **hnswdb** | achieved | 0.0286 | 34,133 | 83,018 | 99,926 |
| B_s01 | 0.95 | **hnswdb** | achieved | 0.0448 | 21,849 | 67,998 | 80,466 |
| B_s01 | 0.98 | **hnswdb** | achieved | 0.0585 | 16,718 | 60,324 | 70,659 |
| B_s01 | 0.99 | **hnswdb** | achieved | 0.0640 | 15,291 | 57,963 | 67,664 |
| B_s01 | 0.90 | **hnswlib** | achieved | 0.0372 | 24,033 | 84,949 | 111,715 |
| B_s01 | 0.95 | **hnswlib** | achieved | 0.0456 | 20,644 | 53,412 | 78,067 |
| B_s01 | 0.98 | **hnswlib** | achieved | 0.0516 | 18,844 | 40,433 | 62,963 |
| B_s01 | 0.99 | **hnswlib** | achieved | 0.0538 | 18,279 | 36,850 | 58,608 |
| fashion_mnist | 0.90 | **faiss** | achieved | 0.1098 | 8,357 | 21,351 | 21,529 |
| fashion_mnist | 0.95 | **faiss** | achieved | 0.1418 | 6,550 | 15,128 | 15,676 |
| fashion_mnist | 0.98 | **faiss** | achieved | 0.1962 | 4,805 | 9,758 | 10,470 |
| fashion_mnist | 0.99 | **faiss** | achieved | 0.2186 | 4,333 | 8,432 | 9,152 |
| fashion_mnist | 0.90 | **hnswdb** | achieved | 0.1027 | 9,071 | 27,599 | 33,988 |
| fashion_mnist | 0.95 | **hnswdb** | achieved | 0.1319 | 7,073 | 19,681 | 24,457 |
| fashion_mnist | 0.98 | **hnswdb** | achieved | 0.1853 | 5,049 | 12,450 | 15,661 |
| fashion_mnist | 0.99 | **hnswdb** | achieved | 0.2075 | 4,513 | 10,687 | 13,499 |
| fashion_mnist | 0.90 | **hnswlib** | achieved | 0.1113 | 8,577 | 25,217 | 28,458 |
| fashion_mnist | 0.95 | **hnswlib** | achieved | 0.1539 | 6,136 | 17,881 | 20,399 |
| fashion_mnist | 0.98 | **hnswlib** | achieved | 0.2379 | 3,910 | 11,262 | 13,037 |
| fashion_mnist | 0.99 | **hnswlib** | achieved | 0.2750 | 3,365 | 9,654 | 11,230 |
<!-- END GENERATED -->

---

### 2.6 Predicate-Filtered Search Benchmark (ACORN Filtering vs NumPy Brute Force)

Evaluates predicate-filtered search across selectivity thresholds ($0.1\%$ to $100\%$) on 10,000 vectors (64d, $M=16, ef_c=100$) vs exact NumPy brute force filtering baseline:

<!-- BEGIN GENERATED table=filtered_search run_id=filtered_20261010_180154_e2247004 -->
**Dataset**: unknown (? x ?d) | **M**=? | **efC**=? | **k**=?

| Selectivity | ef_search | Recall@10 | HNSWDB p50 (ms) | NumPy BF p50 (ms) | HNSWDB Seq QPS | 4T Batch QPS |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.1% (0.1pct) | 50 | 1.0000 | 0.0117 | 0.0150 | 79,259 | 112,887 |
| 0.1% (0.1pct) | 100 | 1.0000 | 0.0115 | 0.0150 | 85,056 | 132,668 |
| 0.1% (0.1pct) | 200 | 1.0000 | 0.0080 | 0.0150 | 114,477 | 129,062 |
| 1.0% (1pct) | 50 | 1.0000 | 1.7422 | 0.0207 | 527 | 656 |
| 1.0% (1pct) | 100 | 1.0000 | 2.3566 | 0.0207 | 416 | 640 |
| 1.0% (1pct) | 200 | 1.0000 | 1.7479 | 0.0207 | 534 | 655 |
| 5.0% (5pct) | 50 | 0.9946 | 0.8276 | 0.0720 | 973 | 2,638 |
| 5.0% (5pct) | 100 | 0.9976 | 1.2006 | 0.0720 | 782 | 2,066 |
| 5.0% (5pct) | 200 | 0.9978 | 1.8724 | 0.0720 | 509 | 1,559 |
| 10.0% (10pct) | 50 | 0.9870 | 0.6207 | 0.1335 | 1,454 | 3,980 |
| 10.0% (10pct) | 100 | 0.9968 | 0.7355 | 0.1335 | 1,281 | 2,976 |
| 10.0% (10pct) | 200 | 0.9988 | 1.4295 | 0.1335 | 661 | 2,200 |
| 25.0% (25pct) | 50 | 0.9672 | 0.4353 | 0.5176 | 1,965 | 5,398 |
| 25.0% (25pct) | 100 | 0.9934 | 0.4655 | 0.5176 | 2,032 | 3,941 |
| 25.0% (25pct) | 200 | 0.9972 | 1.0093 | 0.5176 | 972 | 3,099 |
| 50.0% (50pct) | 50 | 0.9142 | 0.2522 | 1.8299 | 3,547 | 7,058 |
| 50.0% (50pct) | 100 | 0.9740 | 0.3127 | 1.8299 | 2,870 | 5,718 |
| 50.0% (50pct) | 200 | 0.9950 | 0.4808 | 1.8299 | 1,931 | 4,301 |
| 100.0% (100pct) | 50 | 0.8280 | 0.0615 | 2.6945 | 15,918 | 26,298 |
| 100.0% (100pct) | 100 | 0.9360 | 0.1080 | 2.6945 | 8,996 | 14,149 |
| 100.0% (100pct) | 200 | 0.9826 | 0.1881 | 2.6945 | 4,971 | 7,707 |
<!-- END GENERATED -->

