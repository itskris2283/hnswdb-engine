# HNSWDB vs. FAISS & hnswlib: Architectural Trade-offs & Empirical Analysis

## 1. Executive Summary

HNSWDB was engineered from first principles in C++17 as an educational and production-grade ANN vector database. Across synthetic and real-world datasets, HNSWDB achieves **97.0% – 99.9% Recall@10**, sub-millisecond search latency, and consistent throughput across sequential and multi-threaded queries.

In direct comparison with industry benchmarks (**FAISS** and **hnswlib**), HNSWDB demonstrates competitive search latency and memory efficiency, while exhibiting defensible differences in index build throughput and SIMD specialization.

---

## 2. Quantitative Performance Comparison

Evaluated on **Fashion-MNIST** ($N=60,000, D=784, M=16, ef_c=100, k=10$) on Windows 11 x86_64 (AVX2+FMA kernel, MSVC 2022 Release /O2):

| Metric | HNSWDB (This Project) | FAISS HNSW | hnswlib | Analysis & Reason |
| :--- | :---: | :---: | :---: | :--- |
| **Recall@10 ($ef=50$)** | 0.9972 | 0.9948 | 0.9950 | Algorithm 1/2/4 parity with heuristic pruning |
| **Recall@10 ($ef=100$)** | 0.9987 | 0.9978 | 0.9978 | High graph connectivity after deduplication fix |
| **Recall@10 ($ef=200$)** | 0.9991 | 0.9990 | 0.9992 | Full parity on continuous manifold data |
| **Approximation Ratio ($k=10$)** | 1.000 | 1.000 | 1.000 | True Euclidean distance ratio |
| **Sequential Latency (p50)** | 0.220 ms | 0.195 ms | 0.251 ms | Fast greedy graph descent with AVX2+FMA |
| **Build Time (60,000 vectors)** | 20.02 s | 9.66 s | 8.51 s | hnswlib contiguous memory layout & template SIMD |
| **Index Size on Disk** | 187.85 MB | 187.70 MB | 187.94 MB | Compact layer 0 + upper links format |
| **Process RSS Delta** | 194.09 MB | 189.11 MB | 203.17 MB | Compact flat buffer allocations |

---

## 3. Deep Dive: Where & Why FAISS / hnswlib Outperform HNSWDB

### 3.1 Memory Layout (Interleaved vs. Split Flat Arrays)
- **hnswlib:** Stores each element's vector data and link headers in a contiguous memory block (`char* data_level0_memory_`). Traversing a node fetches its links and vector in a single cache line.
- **HNSWDB:** Maintains separate `vectors_` and `layer0_links_` flat arrays. Traversing a node requires two independent memory accesses (one for vector distance computation, one for link traversal), increasing L3/DRAM cache miss frequency during graph traversal.

### 3.2 SIMD Kernel Dispatch vs. Template Specialization
- **FAISS & hnswlib:** Compile dimension-specialized template instantiations (e.g. `L2SqrSIMD16Ext`), allowing compiler loop unrolling and zero-overhead direct inlining.
- **HNSWDB:** Implements dynamic vector dimensions with runtime CPUID kernel dispatch (`DistFunc dist_func_`) and scalar tail loops. As measured in [`results/raw/kernel_microbench.json`](file:///d:/MERN/CustomVecDB/results/raw/kernel_microbench.json), HNSWDB AVX2+FMA kernels achieve high throughput:
  - **16-dim:** 8.0 ns/eval (124.6 M-evals/s)
  - **32-dim:** 7.5 ns/eval (132.5 M-evals/s)
  - **64-dim:** 9.0 ns/eval (111.4 M-evals/s)
  - **128-dim:** 12.7 ns/eval (78.8 M-evals/s)
  - **784-dim:** 53.2 ns/eval (18.8 M-evals/s, 44.2 GFLOPS)
  However, calling distance functions via function pointer introduces minor call indirection relative to direct inlined templates.

### 3.3 Lock Granularity During Parallel Graph Construction
- **hnswlib:** Uses fine-grained `std::mutex` per node array.
- **FAISS:** Uses OpenMP static work-sharing loops for batch operations.
- **HNSWDB:** Uses 1,024 striped spinlocks and a dedicated `std::mutex` for label mapping (`label_to_idx_`). Striped spinlocks keep latency low, but parallel insertion throughput experiences lock contention when thread count exceeds physical core count.

---

## 4. Where HNSWDB Excels Over FAISS / hnswlib

1. **Embedded Metadata Filtering:** FAISS requires external ID selector callbacks; HNSWDB includes an integrated JSON-like AST engine with inverted index acceleration, automatically switching between exact subset brute-force ($\le 2\%$) and expanded graph traversal ($> 2\%$).
2. **Crash-Safe Persistence:** HNSWDB includes IEEE 802.3 CRC32 checksum trailers and atomic write-rename guarantees (`.hdb`), catching silent bit flips and partial writes.
3. **Ergonomic Full-Stack Ecosystem:** Includes native Python bindings (zero-copy NumPy buffers, GIL release), FastAPI REST server with collection management, and Typer CLI.
