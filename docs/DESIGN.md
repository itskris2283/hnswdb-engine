# HNSWDB Design Document

## 1. Architectural Overview

HNSWDB is an approximate nearest neighbor (ANN) vector database engine implemented from scratch in C++17 with Python bindings (`pybind11`), a FastAPI REST service, and a Typer CLI.

The design strictly implements the Hierarchical Navigable Small World (HNSW) graph algorithm formulated by Yu. A. Malkov and D. A. Yashunin (2018), with production enhancements including epoch-stamped visited pools, striped lock concurrency, tombstoning, and hybrid filtered search.

```
+-------------------------------------------------------------------+
|                        Client Applications                        |
|           (Python API, REST API, CLI, C++ Consumers)              |
+-------------------------------------------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
|                      hnswdb / _core (pybind11)                    |
+-------------------------------------------------------------------+
        |                                                   |
        v                                                   v
+-------------------------------+           +-------------------------------+
|          HNSWIndex            |           |           FlatIndex           |
| (Multi-layer Small World)     |           |  (Exact Brute-force Baseline) |
+-------------------------------+           +-------------------------------+
        |               |                                   |
        v               v                                   v
+---------------+ +---------------+                 +---------------+
| MetadataStore | |  VisitedPool  |                 |DistanceKernel |
| (AST + Index) | | (Epoch Ring)  |                 | (AVX2 + FMA)  |
+---------------+ +---------------+                 +---------------+
```

---

## 2. Concurrency Model

### 2.1 Reader-Writer Structure Lock
- A `std::shared_mutex structure_lock_` protects the structural layout.
- Search queries and inserts acquire a `std::shared_lock` (shared reader lock).
- Index resizing (`resize`), graph compaction (`compact`), serialization (`save`), and deserialization (`load`) acquire a `std::unique_lock` (exclusive writer lock).

### 2.2 Striped Node Locks
- Link modifications for individual nodes (Algorithm 1 / Algorithm 4 neighbor selection and link pruning) are protected by a striped array of 1,024 spinlocks (`striped_locks_`).
- Node $i$ maps to lock stripe `i % 1024`.
- When updating connections for node $u$ and neighbor $v$, locks are acquired in sorted order to guarantee deadlock freedom:
  $$\text{lock}(\min(u, v) \pmod{1024}); \quad \text{lock}(\max(u, v) \pmod{1024})$$

### 2.3 Label Hash Map Protection
- External 64-bit integer IDs (`label_t`) are mapped to internal dense indices (`idx_t`) using `std::unordered_map<label_t, idx_t>`.
- A dedicated `std::mutex label_map_mutex_` guards all insertions, lookups, and deletions on the hash map, preventing hash-table rehash data races under concurrent multi-threaded inserts.

---

## 3. Distance Kernels & SIMD Acceleration

Distance computation is the primary inner-loop operation in ANN graph traversal. HNSWDB implements dual kernels with automatic runtime CPUID dispatch:

1. **AVX2 + FMA Kernel:**
   - Processes 8 `float32` elements per vector register using `_mm256_loadu_ps`, `_mm256_fmadd_ps`, `_mm256_sub_ps`, and `_mm256_mul_ps`.
   - Uses 4-way register accumulation unrolling to maximize CPU vector pipeline throughput.
   - Horizontal reduction combines 256-bit registers into a scalar float using vector shuffling.
2. **Scalar Portable Fallback:**
   - Unrolled 4-wide scalar loop handling any hardware architecture or legacy CPU.
3. **Tail Loop:**
   - Arbitrary vector dimensions execute SIMD vector chunks followed by a remainder scalar tail.
4. **Software Prefetching:**
   - The greedy search loop issues `_mm_prefetch` (`_MM_HINT_T0`) on the neighbor vector's memory block before distance calculation.

---

## 4. Memory Layout & Cache Optimization

All node vectors are stored in a contiguous buffer:
- Node vectors: `float vectors_[max_elements * dim]`.
- Layer 0 graph links: Flattened contiguous array `layer0_links_[max_elements * (1 + 2*M)]` with fixed stride `(1 + 2*M)` where index 0 stores the degree count followed by link target internal IDs.
- Upper layers ($l \ge 1$): Dynamically allocated only for the minority of nodes where $\text{level} > 0$ ($\approx 1/M$ fraction of nodes), dramatically reducing memory footprint.
- Parallel flat vectors:
  - `labels_`: `std::vector<label_t>` (8 bytes/element)
  - `levels_`: `std::vector<uint8_t>` (1 byte/element)
  - `deleted_`: `std::vector<uint8_t>` (1 byte tombstone flag per internal node)

---

## 5. Epoch-Based Visited List Pool

Graph traversal requires tracking visited nodes to prevent cycles:
- Standard `std::unordered_set<idx_t>` incurs dynamic heap allocation on every search ($O(ef)$ mallocs), severely degrading search QPS.
- HNSWDB implements a preallocated `VisitedListPool`. Each thread checks out a reusable buffer of `uint16_t` tags.
- Instead of clearing the array ($O(N)$ operations), an integer `cur_epoch_` is incremented ($O(1)$ reset). A node $u$ is visited if `tag[u] == cur_epoch_`.
- When `cur_epoch_` overflows $2^{16} - 1$, the buffer is zeroed with `memset` and `cur_epoch_` resets to 1.

---

## 6. Filtered Search Strategy

HNSWDB supports expressive JSON/AST metadata queries (`$eq`, `$ne`, `$gt`, `$gte`, `$lt`, `$lte`, `$in`, `$and`, `$or`, `$not`) via inverted index acceleration:
1. **Selectivity Estimation:**
   - The query AST evaluates matching candidate count using inverted postings lists (`field -> value -> sorted vector<idx_t>`).
2. **Hybrid Execution Path:**
   - **Highly Selective ($\le 0.02$ match fraction or $\le 5,000$ candidates):** Falls back to exact brute-force search over the filtered subset. Graph search in disconnected or sparse subgraphs suffers from topological disconnects; exact subset scanning is faster and delivers exact recall.
   - **Mildly Selective ($> 0.02$ match fraction):** Executes HNSW graph traversal with dynamic expansion: $ef_{\text{effective}} = \min(ef / \text{selectivity}, 10 \cdot ef)$. Non-matching nodes are traversed to maintain connectivity, but only matching nodes enter the candidate result set.
