# Known Limitations & Performance Characteristics

This document transparently records verified engineering limitations, dataset sensitivities, and trade-offs in HNSWDB.

---

## 1. Clustered Distribution Sensitivity & $ef_{construction}$ Dependency

- **Symptom:** On synthetic distributions with tight, well-separated Gaussian clusters (such as 50 clusters with standard deviation $\sigma=0.1$ across 20,000 vectors in 64 dimensions, Dataset B), HNSW graph construction with standard $ef_{construction}=100$ reaches a recall ceiling around **0.9716** at $ef_{search}=200$.
- **Mechanism:** As proven in [INVESTIGATIONS.md](INVESTIGATIONS.md), tight clusters cause **505 nodes (2.5%) to become search-unreachable** (failing 1-NN self-retrieval). During graph construction, nodes in the same tight cluster have near-identical distances to each other. When neighbor lists overflow ($> M_0$), `select_neighbors_heuristic` aggressively prunes intra-cluster connections. Over thousands of insertions, early-inserted nodes lose their incoming connections, rendering them unreachable by greedy search. In comparison, `hnswlib` has only 7 unreachable nodes on the identical dataset.
- **Resolution / Workarounds:**
  - Increasing $ef_{construction}$ to **400** provides sufficient candidate exploration during build to prevent link starvation, yielding **0 unreachable nodes** and **1.0000** recall on the exact same dataset.
  - On cluster spreads of $\sigma \ge 0.3$, HNSWDB achieves **1.0000** recall@10 even at $ef_{construction}=100$.
  - On real-world continuous manifold datasets (such as Fashion-MNIST $784$-d), the standard $ef_{construction}=100$ achieves **0.9996** recall@10 at $ef_{search}=200$.

---

## 2. Build-Time Throughput Relative to C++ Template Specializations

- **Observation:** HNSWDB single-thread index construction is slower than `hnswlib` and `FAISS` on high-dimensional vectors.
- **Root Causes:**
  1. **Memory Layout:** `hnswlib` packs node vectors and link headers contiguously in memory (`data_level0_memory_`), allowing vector data and links to be loaded within the same cache line. HNSWDB stores vector data and layer-0 links in distinct flat buffers (`vectors_` and `layer0_links_`), incurring two separate DRAM/L3 cache fetches per traversed node.
  2. **Runtime Kernel Dispatch:** HNSWDB uses runtime function pointer dispatch (`DistFunc dist_func_`) to select between AVX2 and scalar paths, which prevents loop unrolling across the distance function boundary. Reference libraries instantiate dimension-specialized templates (e.g. `L2SqrSIMD16Ext`) at compile time.

---

## 3. Tombstone Garbage Collection & Compaction

- **Behavior:** Deleted vectors are tombstoned via bit flags in `deleted_` (`uint8_t` array) and excluded from query results in $O(1)$.
- **Limitation:** Tombstoned nodes retain their graph link positions until explicit `index.compact()` is executed. In workloads with high churn ($> 30\%$ deletions), calling `compact()` rebuilds a clean graph to reclaim memory and maintain search recall.

---

## 4. Platform and Sanitizer Verification

- **Host Platform:** Development and primary validation were performed on Windows 11 (MSVC 2022 /O2, AVX2+FMA SIMD kernel).
- **Sanitizers:** ASan / UBSan / TSan verification targets require Linux/GCC/Clang execution in CI environments. On native Windows MSVC builds, memory correctness is verified via extensive unit tests, boundary bounds-checking, and zero-leak RAII allocators.
