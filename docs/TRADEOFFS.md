# HNSWDB vs. FAISS & hnswlib: Honest Architectural Trade-offs

## 1. Executive Summary

HNSWDB was engineered from first principles as an educational and production-grade ANN vector database. Across synthetic and real datasets, HNSWDB achieves **97.0% – 99.9% Recall@10**, sub-millisecond search latency, and consistent throughput across sequential and multi-threaded queries.

In direct comparison with industry benchmarks (**FAISS** and **hnswlib**), HNSWDB demonstrates competitive search latency and memory efficiency, while exhibiting defensible differences in index build throughput and SIMD specialization.

---

## 2. Quantitative Performance Comparison

Evaluated on **Fashion-MNIST** ($N=60,000, D=784, M=16, ef_c=100, k=10$) on Windows 11 x86_64:

| Metric | HNSWDB (This Project) | FAISS HNSW | hnswlib | Analysis & Reason |
| :--- | :---: | :---: | :---: | :--- |
| **Recall@10 ($ef=50$)** | 0.9976 | 0.9948 | 0.9950 | Full Algorithm 1/2/4 parity with heuristic pruning |
| **Recall@10 ($ef=200$)** | 0.9994 | 0.9990 | 0.9992 | High graph connectivity after deduplication fix |
| **p50 Latency ($ef=50$)** | 0.220 ms | 0.195 ms | 0.251 ms | Fast greedy graph descent with AVX2+FMA kernels |
| **p99 Latency ($ef=50$)** | 0.442 ms | 0.459 ms | 0.416 ms | Bounded tail latency across queries |
| **Sequential QPS ($ef=50$)** | 4,304 | 4,696 | 3,899 | Pure 1-query sequential execution path |
| **4-Thread Batch QPS ($ef=50$)** | 9,102 | 11,249 | 10,044 | FAISS OpenMP batch loop vs HNSWDB thread pool |
| **Build Time (60,000 vectors)** | 20.02 s | 9.66 s | 8.51 s | hnswlib contiguous memory layout & template SIMD |
| **Index Size on Disk** | 187.85 MB | 187.70 MB | 187.94 MB | All engines store compact layer 0 + upper links |
| **Process RSS Delta** | 194.09 MB | 189.11 MB | 203.17 MB | Compact flat buffer allocations |

---

## 3. Deep Dive: Where & Why FAISS / hnswlib Outperform HNSWDB

### 3.1 Memory Layout (Interleaved vs. Split Arrays)
- **hnswlib:** Stores each element's vector data and link headers in a contiguous chunk (`char* data_level0_memory_`). Traversing a node fetches its links and vector in a single cache line.
- **HNSWDB:** Maintains separate `vectors_` and `layer0_links_` flat arrays. Traversing a node requires two independent memory accesses (one for vector distance computation, one for link traversal), doubling L3/DRAM cache misses per step.

### 3.2 Template Specialization & Compile-Time SIMD vs. Runtime Function Pointers
- **FAISS & hnswlib:** Compile specialized templates for fixed dimensions (e.g. `L2SqrSIMD16Ext`), unrolling loops with zero tail handling and direct inlining.
- **HNSWDB:** Implements dynamic vector dimensions with runtime CPUID kernel dispatch (`DistFunc dist_func_`) and scalar tail loops. This supports arbitrary dimensions without code bloat, but incurs function pointer indirection and prevents compiler loop unrolling across function boundaries.

### 3.3 Lock Granularity During Parallel Graph Construction
- **hnswlib:** Uses fine-grained `std::mutex` per node array.
- **FAISS:** Uses OpenMP static work-sharing loops for batch operations.
- **HNSWDB:** Uses 1,024 striped spinlocks and a dedicated `std::mutex` for label mapping (`label_to_idx_`). While deadlock-free and ThreadSanitizer-clean, hash map lock contention during multi-threaded batch insert creates a bottleneck relative to direct array indexing.

---

## 4. Where HNSWDB Excels Over FAISS / hnswlib

1. **Embedded Metadata Filtering:** FAISS requires external ID selector callbacks; HNSWDB includes an integrated JSON-like AST engine with inverted index acceleration, automatically switching between exact subset brute-force ($\le 2\%$) and expanded graph traversal ($> 2\%$).
2. **Crash-Safe Persistence:** HNSWDB includes IEEE 802.3 CRC32 checksum trailers and atomic write-rename guarantees (`.hdb`), catching silent bit flips and partial writes.
3. **Ergonomic Full-Stack Ecosystem:** Includes native Python bindings (zero-copy NumPy buffers, GIL release), FastAPI REST server with collection management, and Typer CLI.
