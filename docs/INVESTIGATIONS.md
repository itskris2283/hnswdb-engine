# Engineering Investigations & Performance Debugging

This document records empirical investigations, root-cause analyses, and before/after verification for performance and quality anomalies in HNSWDB. Every metric reported here corresponds to raw result artifacts in `results/raw/` registered in [`results/raw/MANIFEST.json`](file:///d:/MERN/CustomVecDB/results/raw/MANIFEST.json).

---

## Investigation 1: Neighbor Selection Deduplication & Graph Invariants

### 1. Context & Symptom
- **Observed Behavior:** In initial test runs, duplicate neighbor candidates could be selected during heuristic edge pruning (`select_neighbors_heuristic`) when `keep_pruned` was active, and reciprocal edge updates could attempt redundant link insertions.
- **Impact:** Redundant candidate evaluations and wasted link slots reduced graph diversity and search recall.

### 2. Diagnostic Methodology
We implemented [`HNSWIndex::diagnostics()`](file:///d:/MERN/CustomVecDB/src/hnsw_index.cpp) and graph topology audits to track:
- **Snapshot graph invariants:** Exact count of duplicate outgoing links (`duplicate_links_count`), self-loops (`self_links_count`), in-degree distribution, weakly connected components (`weakly_connected_components`), and layer-0 BFS reachability from the entry point (`unreachable_nodes_count`).
- **Cumulative runtime counters:** Total calls to heuristic selection (`cumulative_heuristic_calls`), overflow re-selections (`cumulative_overflow_reselections`), and rejected duplicate link insertion attempts (`cumulative_duplicate_insert_attempts`).

> [!NOTE]
> **Clarification on Duplicate Counters:**
> In previous internal drafts, cumulative build-time loop iterations were ambiguously referred to as "duplicate links in the graph". In an $N=5,000$ index with $M_0=32$, the total capacity of layer 0 is at most $5,000 \times 32 = 160,000$ slots. HNSWDB now strictly separates:
> 1. **`duplicate_links_count`**: The exact number of duplicate neighbor IDs present inside the graph's serialized adjacency lists (which is strictly 0).
> 2. **`cumulative_duplicate_insert_attempts`**: Cumulative runtime occurrences of duplicate candidates rejected during dynamic edge additions.

### 3. Root Cause Analysis
In `select_neighbors_heuristic`:
1. When candidates were sorted and pruned, rejected candidates were appended to `discarded_list`. If `return_list` had remaining capacity, candidates from `discarded_list` were appended without verifying whether they were already present in `return_list`.
2. When reverse links were added to neighbor nodes, duplicate checks were needed to prevent adding existing connections.

### 4. Fix Applied
In [`src/hnsw_index.cpp`](file:///d:/MERN/CustomVecDB/src/hnsw_index.cpp):
1. **Candidate Deduplication:** Unique filtering is applied to candidates prior to beam selection.
2. **Discarded List Uniqueness Guard:** Elements from `discarded_list` are checked against `return_list` before insertion.
3. **Legacy Bug Compile Guard:** Added `-DHNSWDB_LEGACY_DUPLICATE_BUG` in CMake to allow reproducible before/after verification on the identical codebase and commit.

### 5. Empirical Verification & Measured Results

All runs executed under identical hardware configuration (Intel Core i7-8650U, AVX2+FMA kernel, MSVC 2022 Release /O2). Raw JSON artifacts:
- Before fix: [`results/raw/bug_before_dataset_a.json`](file:///d:/MERN/CustomVecDB/results/raw/bug_before_dataset_a.json), [`results/raw/bug_before_dataset_b.json`](file:///d:/MERN/CustomVecDB/results/raw/bug_before_dataset_b.json), [`results/raw/bug_before_fashion_mnist.json`](file:///d:/MERN/CustomVecDB/results/raw/bug_before_fashion_mnist.json)
- After fix: [`results/raw/bug_after_dataset_a.json`](file:///d:/MERN/CustomVecDB/results/raw/bug_after_dataset_a.json), [`results/raw/bug_after_dataset_b.json`](file:///d:/MERN/CustomVecDB/results/raw/bug_after_dataset_b.json), [`results/raw/bug_after_fashion_mnist.json`](file:///d:/MERN/CustomVecDB/results/raw/bug_after_fashion_mnist.json)

#### Dataset A: Clustered Synthetic ($N=5,000, D=32, M=16, ef_c=100, k=10$)
| $ef_{search}$ | Before Fix (`HNSWDB_LEGACY_DUPLICATE_BUG=ON`) | After Fix (Current Clean Engine) | Delta |
| :--- | :---: | :---: | :---: |
| 10 | 0.9803 | **0.9822** | +0.19% |
| 20 | 0.9937 | **0.9957** | +0.20% |
| 50 | 0.9953 | **0.9972** | +0.19% |
| 100 | 0.9953 | **0.9972** | +0.19% |
| 200 | 0.9953 | **0.9972** | +0.19% |
| **Self-Retrieval Rate (1-NN)** | 0.9970 | **0.9990** | +0.20% |
| **Unreachable Nodes (BFS)** | 12 | **9** | -25.0% |
| **Snapshot Duplicate Links** | 0 | **0** | Invariant verified |

#### Dataset B: Clustered Synthetic ($N=20,000, D=64, M=16, ef_c=100, k=10$)
| $ef_{search}$ | Before Fix | After Fix | Delta |
| :--- | :---: | :---: | :---: |
| 10 | 0.9080 | **0.9090** | +0.10% |
| 20 | 0.9585 | **0.9591** | +0.06% |
| 50 | 0.9696 | **0.9702** | +0.06% |
| 100 | 0.9701 | **0.9706** | +0.05% |
| 200 | 0.9702 | **0.9707** | +0.05% |
| **Self-Retrieval Rate (1-NN)** | 0.9550 | **0.9550** | 0.00% |

#### Dataset C: Fashion-MNIST ($N=60,000, D=784, M=16, ef_c=100, k=10$)
| $ef_{search}$ | Before Fix | After Fix | Delta |
| :--- | :---: | :---: | :---: |
| 10 | 0.9697 | **0.9698** | +0.01% |
| 20 | 0.9894 | **0.9894** | 0.00% |
| 50 | 0.9972 | **0.9972** | 0.00% |
| 100 | 0.9987 | **0.9987** | 0.00% |
| 200 | 0.9991 | **0.9991** | 0.00% |
| **Self-Retrieval Rate (1-NN)** | 0.9550 | **0.9630** | +0.80% |

- **Regression Test:** [`tests/python/test_regression_bug.py`](file:///d:/MERN/CustomVecDB/tests/python/test_regression_bug.py) verifies graph connectivity, zero duplicate links, and recall $\ge 0.995$ on Dataset A.

---

## Investigation 2: Build-Time Performance & Multi-threaded Scaling

### 1. Context & Symptom
Index construction in HNSWDB takes longer than `hnswlib` (e.g. single-thread build time ratio is ~2.5x vs hnswlib on synthetic workloads).

### 2. Hotspots Identified
1. **Dynamic Memory Allocation in Neighbor Pruning:** Frequent heap allocations during heuristic shrinking.
2. **Lock Hold Duration in Graph Search:** Copying neighbor lists under spinlocks during multi-threaded batch ingestion.

### 3. Optimizations Applied
In [`src/hnsw_index.cpp`](file:///d:/MERN/CustomVecDB/src/hnsw_index.cpp):
1. **Stack Buffer Neighbor Copy:** `search_layer` copies links into a fixed stack array (`uint32_t local_nbrs[512]`) under the spinlock, releasing the lock in $< 20$ ns before executing distance calculations and prefetch instructions lock-free.
2. **In-place Pruning:** Removed redundant allocations in candidate selection.

### 4. Measured Impact
On $N=5,000, D=32, M=16, ef_c=100$, single-thread index construction completed in **1.16 s** with **17.5M distance computations** (~3,500 distance computations per inserted vector).

### 5. Architectural Comparison with hnswlib and FAISS
1. **Memory Layout:** `hnswlib` packs vector bytes and link offsets into a single contiguous structure (`data_level0_memory_`), allowing link and vector data to share cache lines. HNSWDB stores vectors in `vectors_` (`std::vector<float>`) and links in `layer0_links_` (`std::vector<uint32_t>`), resulting in separate cache accesses per node.
2. **SIMD Kernel Specialization:** `hnswlib` and FAISS generate compile-time template instantiations for standard dimensions (e.g., $d=16, 32, 64, 128$), enabling full loop unrolling. HNSWDB uses runtime function pointer dispatch (`DistFunc dist_func_`), adding minor function call indirection per distance evaluation.
