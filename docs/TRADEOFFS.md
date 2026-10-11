# HNSWDB vs. FAISS & hnswlib: Architectural Trade-offs & Empirical Analysis

## 1. Executive Summary

On Fashion-MNIST and on the clustered synthetic set (after the construction defaults were corrected), HNSWDB's recall matches FAISS and hnswlib at equal parameters. Search speed is comparable, and the ordering between engines is not stable across runs on this host (relative speed varies between runs on this host; treat as parity). A small number of nodes remain unreachable on clustered data (see target T5 in REPORT.md).

---

## 2. Quantitative Performance Comparison

Evaluated on **Fashion-MNIST** ($N=60,000, D=784, M=16, ef_c=100, k=10$) on Windows 11 x86_64 (AVX2+FMA kernel, MSVC 2022 Release /O2):

<!-- BEGIN GENERATED table=summary_fashion_mnist run_id=fashion_mnist_20261010_174828_0a9f92db -->
**Dataset**: fashion_mnist (60000 x 784d) | **M**=16 | **efC**=100 | **k**=10

| Metric | HNSWDB | FAISS HNSW | hnswlib | Analysis & Notes |
| :--- | :---: | :---: | :---: | :--- |
| **Recall@10 (ef=50)** | 0.9951 | 0.9948 | 0.9951 | Competitive parity |
| **p50 Latency (ef=50)** | 0.220 ms | 0.230 ms | 0.296 ms | Sub-millisecond graph search |
| **Build Time** | 29.94 s | 28.23 s | 28.42 s | Single-thread build |
| **Process RSS Delta** | 194.8 MB | 188.9 MB | 201.9 MB | Memory footprint |
<!-- END GENERATED -->

---

## 3. Deep Dive: Architectural Comparisons

### 3.1 Build-Time Performance

Build-time behavior is reported only through the generated tables in BENCHMARKS.md and INVESTIGATIONS.md. See KNOWN_LIMITATIONS.md section 2.

### 3.2 SIMD Distance Kernel Microbenchmark

Distance computation throughput measured across vector dimensions on host CPU (AVX2+FMA):

<!-- BEGIN GENERATED table=kernel_microbench run_id=kernel_microbench -->
**SIMD Kernel**: avx2 (AVX2+FMA)

| Dimension | Latency (ns/eval) | Throughput (M-evals/s) | Effective GFLOPS |
| ---: | ---: | ---: | ---: |
| 16 | 8.02 | 124.65 | 5.98 |
| 32 | 7.55 | 132.52 | 12.72 |
| 64 | 8.98 | 111.38 | 21.38 |
| 128 | 12.69 | 78.79 | 30.26 |
| 784 | 53.22 | 18.79 | 44.20 |
<!-- END GENERATED -->

### 3.3 Lock Granularity During Parallel Graph Construction

- **hnswlib:** Uses fine-grained `std::mutex` per node array.
- **FAISS:** Uses OpenMP static work-sharing loops for batch operations.
- **HNSWDB:** Uses 1,024 striped spinlocks and a dedicated `std::mutex` for label mapping (`label_to_idx_`).

---

## 4. Where HNSWDB Excels Over FAISS / hnswlib

1. **Embedded Metadata Filtering:** Unlike libraries that rely on external callback wrappers, HNSWDB includes an integrated JSON-like AST engine with inverted index acceleration, automatically switching between exact subset scan ($\le 2\%$) and expanded graph traversal ($> 2\%$).
2. **Crash-Safe Persistence:** HNSWDB includes IEEE 802.3 CRC32 checksum trailers and atomic write-rename guarantees (`.hdb`), catching silent bit flips and partial writes.
3. **Ergonomic Full-Stack Ecosystem:** Includes native Python bindings (zero-copy NumPy buffers, GIL release), FastAPI REST server with collection management, and Typer CLI.
