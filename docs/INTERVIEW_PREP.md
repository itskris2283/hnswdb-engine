# HNSWDB: Technical Interview Preparation & Architecture Guide

This guide is designed for technical deep-dive interviews in **AI Infrastructure, Systems Engineering, and High-Performance Backend Engineering**.

---

## 1. The Two-Minute Executive Pitch

> **"What is HNSWDB?"**
> "HNSWDB is an approximate nearest neighbor (ANN) vector database engineered from scratch in C++17 with Python bindings (`pybind11`), a FastAPI REST server, and a Typer CLI.
>
> I built it to understand and master the systems challenges of vector search: low-level SIMD intrinsics, memory-hierarchy cache locality, concurrent graph synchronization, and crash-safe storage.
>
> On standard benchmarks like **Fashion-MNIST (784 dimensions, 60,000 vectors)**, HNSWDB achieves **0.9991 Recall@10** at p50 search latency under 0.3 ms at $ef=50$. In head-to-head empirical evaluations against reference libraries like **FAISS** and **hnswlib**, HNSWDB delivers matching search accuracy and competitive query throughput, while incorporating built-in features that standalone index libraries lack: **hybrid inverted-index metadata filtering**, **atomic CRC32 persistence (`.hdb`)**, and **zero-copy Python integration**."

---

## 2. Core Architecture & System Design Decisions

```
+-------------------------------------------------------------------------+
|                           Client Layer                                  |
|            Python API (Zero-Copy NumPy)  |  FastAPI REST  |  CLI        |
+-------------------------------------------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
|                  C++17 Engine Core (hnswdb::_core)                      |
+-------------------------------------------------------------------------+
       |                           |                          |
       v                           v                          v
+---------------+          +----------------+         +-------------------+
| Distance (SIMD|          |   HNSW Graph   |         | Metadata & Filter |
| AVX2+FMA/CPUID|          | 1024 Striped   |         | Inverted Postings |
| Unrolled Loop |          | Spinlocks +    |         | Dynamic Hybrid    |
| Tail Remainder|          | Epoch Tag Pool |         | Brute/Graph Path  |
+---------------+          +----------------+         +-------------------+
```

### Decision Matrix & Trade-offs

| Component | Design Choice in HNSWDB | Alternative Considered | Engineering Rationale & Trade-off |
| :--- | :--- | :--- | :--- |
| **Distance Dispatch** | Runtime CPUID dispatch to AVX2+FMA with unrolled scalar fallback | Compile-time template specialization per dimension (hnswlib approach) | Runtime dispatch supports arbitrary dynamic dimensions (e.g. $D=99, 960$) without binary bloat; incurs minor function pointer indirection during inner loops. |
| **Visited Nodes** | Epoch-based Tag Pool (`uint16_t` epoch ring) | `std::unordered_set<idx_t>` or dynamic `std::vector<bool>` | Resetting `unordered_set` requires $O(ef)$ heap allocations per query; resetting bitsets requires $O(N)$ memory writes. Epoch tag pool achieves $O(1)$ reset by incrementing a counter. |
| **Concurrency** | 1,024 striped spinlocks + Reader-Writer `std::shared_mutex` | Global mutex OR per-node `std::mutex` | Reader-writer lock allows concurrent reader searches while guarding resize/compaction. Striped spinlocks prevent false sharing without consuming 40 bytes of `std::mutex` per node. |
| **Memory Layout** | Split flat arrays (`vectors_` contiguous float array, `layer0_links_` contiguous array) | Interleaved buffer per node (`char* data_level0_memory_`) | Split arrays allow zero-copy vector loading and straightforward NumPy memory-mapping; requires two cache-line fetches during neighbor traversal rather than one. |
| **Metadata Filter** | Selectivity estimation: exact subset scan if $\le 2\%$, graph expansion if $> 2\%$ | Post-filtering only OR pre-filtering only | Post-filtering fails when selectivity is low ($< 1\%$ match rate yields empty results); pre-filtering on graphs suffers from graph fragmentation. Hybrid path guarantees high recall and throughput. |
| **Persistence** | Little-endian binary with 64-byte header, section blocks, CRC32 trailer, atomic write-rename (`.hdb`) | Unchecked raw dump OR Protobuf/JSON | Binary stream maximizes read/write speed; CRC32 detects silent bit rot; atomic rename prevents corrupted indices on unexpected crashes. |

---

## 3. Deep-Dive Debugging Narratives (STAR Stories)

### Story 1: The Neighbor Deduplication & Graph Invariant Investigation
- **Situation:** During high-$ef$ accuracy benchmarking, candidate selection during heuristic edge pruning had edge cases where rejected candidates in `discarded_list` were reintroduced without deduplication.
- **Task:** Build comprehensive diagnostics in `HNSWIndex::diagnostics()` to track snapshot graph properties and guarantee graph integrity.
- **Action:** Implemented diagnostics tracking degree distributions, reachability, zero-in-degree nodes, and duplicate links. Verified that snapshot duplicate links are strictly 0 and added deduplication guards during heuristic selection.
- **Result:** Graph integrity verified across all layers; Fashion-MNIST recall reached **0.9991**, achieving parity with reference baselines. Added a permanent regression test `test_regression_bug.py`.

### Story 2: Profiling the Index Build-Time Gap
- **Situation:** HNSWDB build time on 10,000 vectors was $8.36\text{s}$ (1T) and $5.34\text{s}$ (8T), lagging hnswlib ($2.58\text{s}$ 1T, $0.49\text{s}$ 8T).
- **Task:** Profile the build pipeline to eliminate bottlenecks and explain the remaining gap.
- **Action:** Instrumenting internal stages identified that `select_neighbors_heuristic` allocated `std::unordered_set` hash tables on every link update (over 350,000 heap allocations per build), while `search_layer` held striped spinlocks during memory copies and visited checks.
- **Optimization:** Replaced heap hashtables with stack-based sorting and linear uniqueness scanning; reduced spinlock hold time in `search_layer` using small stack buffers.
- **Result:** Build time dropped from $8.36\text{s} \to 5.54\text{s}$ (1T) and $5.34\text{s} \to 2.12\text{s}$ (8T) — a **$2.6\times$ parallel scaling improvement**. Documented remaining architectural differences (contiguous node interleaving and template SIMD loop unrolling).

### Story 3: The Single-Query Latency Artifact
- **Situation:** Early benchmarks suggested HNSWDB was 3x faster than FAISS on single-query p50 latency (0.056 ms vs 0.162 ms), yet FAISS had higher batch QPS.
- **Task:** Investigate the mathematical contradiction ($p50 = 0.162\text{ ms}$ implies $\le 6,170\text{ QPS}$, yet FAISS achieved $18,740\text{ QPS}$).
- **Action:** Investigated the benchmark harness and discovered that FAISS was being called via `faiss.search` with default multi-threading enabled without calling `faiss.omp_set_num_threads(1)`, incurring OpenMP thread synchronization overhead on single-row slices.
- **Result:** Standardized test harness call paths: isolated 1-query slices with strict per-thread controls (`omp_set_num_threads(1)`), separated sequential single-thread QPS from multi-thread batch QPS, and added automated validation (`|1000/mean_lat - sequential_qps| / sequential_qps <= 0.05`).

---

## 4. Key Technical Questions & Answers

### Q1: "Why use HNSW instead of IVF-PQ or ScaNN?"
> "HNSW offers superior recall-latency Pareto efficiency for in-memory indices where recall $> 95\%$ is required. IVF-PQ (Inverted File with Product Quantization) compresses massive datasets to fit limited RAM, but vector quantization introduces quantization error that bounds the recall ceiling. ScaNN uses anisotropic vector quantization optimized for MIPS on CPU/TPU. For low-latency, exact-vector representation with $\ge 99\%$ recall, HNSW graph navigation provides high accuracy."

### Q2: "How does HNSW handle dynamic insertions and deletions without corrupting the graph?"
> "Insertions use striped spinlocks to update adjacency lists in sorted order, preventing deadlocks. Deletions use tombstone flags in `deleted_` (`uint8_t` array) for $O(1)$ exclusion from query results. When tombstones exceed a threshold (e.g. 20%), `compact()` creates a clean, densely indexed graph."
