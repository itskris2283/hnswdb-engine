# HNSWDB: Technical Interview Preparation & Architecture Deep-Dive

This reference is designed for technical interviews in **Systems Engineering, AI Infrastructure, and High-Performance Backend Engineering**. Every claim is grounded in the C++17 implementation and reproducible benchmark runs in `results/raw/`.

---

## 1. The Two-Minute Executive Pitch

> **"What is HNSWDB?"**
>
> "HNSWDB is an approximate nearest neighbor (ANN) vector database engineered from scratch in C++17 with Python bindings (`pybind11`), a FastAPI REST service, and a Typer CLI.
>
> I built it to master the systems challenges of vector search engines: low-level AVX2+FMA SIMD intrinsics with runtime CPUID dispatch, cache locality across large graph structures, concurrent thread synchronization with fine-grained striped spinlocks, and crash-safe binary persistence.
>
> On standard benchmarks like **Fashion-MNIST (784 dimensions, 60,000 vectors)**, HNSWDB achieves competitive Recall@10 at rapid query latency (verified in [BENCHMARKS.md](BENCHMARKS.md)). In empirical head-to-head evaluations against **FAISS** and **hnswlib**, HNSWDB delivers matching search accuracy and query throughput at parity, while integrating full database capabilities that standalone index libraries omit: **hybrid inverted-index metadata filtering**, **atomic IEEE 802.3 CRC32 persistence (`.hdb`)**, and **zero-copy Python integration**."

---

## 2. The Ten-Minute Architecture Deep-Dive

When an interviewer asks for a comprehensive architectural walkthrough (10 minutes), structure the explanation into five concrete layers:

### 1. Vector Storage & Distance Computation Core
- **Layout:** Contiguous flat memory buffer (`std::vector<float> vectors_`). Vectors are assigned monotonic internal dense IDs (`idx_t = uint32_t`) mapped to user labels (`label_t = int64_t`) via hash tables.
- **SIMD Dispatch:** Automatic runtime CPUID detection at startup. If AVX2 and FMA are supported by the host CPU, function pointers point to 8-wide 256-bit float kernels (`_mm256_fmadd_ps`) with 4-way unrolling (32 floats per iteration) and scalar remainder loops. Odd vector dimensions are padded or handled by scalar tail loops.

### 2. Multi-Layer Hierarchical Graph Topology
- **Layer Distribution:** Probabilistic exponential skip-list hierarchy where each node's maximum layer $l$ is drawn from an exponential distribution with scale factor $m_L = 1 / \ln(M)$. Layer 0 contains all vectors; each layer $l > 0$ contains exponentially fewer vectors.
- **Link Storage:** Contiguous arrays for Layer 0 (`layer0_links_`, capacity $M_0 = 2M$ per node) and dynamic adjacency lists for upper layers (`upper_links_[l][u]`, capacity $M$ per node).

### 3. Query Traversal & Graph Beam Search
- **Top-Down Routing:** Greedy descent from top layer $L_{max}$ down to Layer 1. At each upper layer, the search hops greedily ($ef=1$) to the nearest neighbor until a local minimum is reached; that node becomes the entry point for the next layer down.
- **Layer 0 Search:** Switches to beam search with width $ef_{search}$. Maintained using a min-heap candidate queue and a max-heap result set. Visited nodes are tracked using an epoch tag pool (`VisitedListPool`) with $O(1)$ integer increments, completely avoiding heap allocations during search.

### 4. Thread Synchronization & Concurrent Ingestion
- **Two-Tier Synchronization:** 
  1. A `std::shared_mutex structure_lock_`: multiple concurrent readers acquire shared ownership during `search()`, while batch reallocations (`resize()`, `compact()`) acquire exclusive ownership.
  2. Striped node locks: an array of 1,024 spinlocks (`striped_locks_`). When adding edges between node $u$ and neighbor $v$, spinlocks are acquired in strictly sorted order: $\min(u, v) \pmod{1024}$ followed by $\max(u, v) \pmod{1024}$, preventing deadlocks.

### 5. Crash-Safe Binary Serialization (`.hdb`)
- **Format:** Little-endian binary format featuring a 64-byte structured header, contiguous payload blocks (labels, vectors, layer-0 links, upper links, metadata), and a trailing 4-byte IEEE 802.3 CRC32 checksum.
- **Atomic Write-Rename:** Index saves write to a temporary sibling file, flush data with `fflush` and OS-level file buffers, and execute an atomic file rename over the destination path.

---

## 3. Three True Debugging Stories (STAR Method)

### Story 1: The Benchmark Methodology & Approximation Ratio Artifacts
- **Situation:** Early benchmark results reported contradictory single-query latency and batch throughput measurements for FAISS, where reported p50 latency did not match sequential throughput. Concurrently, Fashion-MNIST approximation ratio was reported as ~31.9 instead of ~1.0.
- **Task:** Dissect the benchmarking harness, isolate the measurement discrepancies, and restore mathematical consistency.
- **Action:**
  1. Profiled FAISS query execution and discovered that calling `faiss.search(xq[i:i+1])` without `omp_set_num_threads(1)` spawned OpenMP thread-pool synchronization on every individual 1-row slice, artificially distorting single-query latency.
  2. For the approximation ratio, inspected the dataset loader: `fashion-mnist-784-euclidean.hdf5` stores precomputed Euclidean distances ($d \approx 482$). The metrics function assumed squared Euclidean distances and took an extra square root ($\sqrt{482} \approx 22$), inflating the ratio.
  3. Standardized sequential latency with proper warmup and thread pinning, and added an automated sanity checker (`|1000/mean_lat - QPS| / QPS <= 0.05`).
- **Result:** Both anomalies were eliminated; approximation ratios correctly evaluated to 1.000, and latency/throughput metrics reconciled across all engines.

### Story 2: The Falsified Duplicate-Links Hypothesis
- **Situation:** During early investigations of a recall gap on clustered data, an external reviewer suspected that duplicate links were saturating neighbor lists. A report prematurely claimed duplicate links were fixed, improving recall from ~0.991 to ~0.999.
- **Task:** Validate whether duplicate links actually existed and whether the change caused the claimed recall jump.
- **Action:**
  1. Formulated mathematical invariants: in a 5,000-node index with $M_0=32$, Layer 0 contains at most $5,000 \times 32 = 160,000$ links. A count of millions was mathematically impossible for a snapshot graph.
  2. Implemented strict invariant test `DuplicateAndSelfLinksZero` in C++, checking every link across all layers. The test verified 0 duplicate links existed in snapshot graphs.
  3. Ran controlled before/after ablations across 5 seeds: deduplicating rejected candidates during overflow backfill produced a recall delta of exactly 0.00 (pure noise).
- **Result:** Retracted the unproven claim completely; built regression tests to permanently prevent unfalsified hypotheses from entering docs; and redirected effort toward isolating the actual construction mechanism.

### Story 3: The Clustered-Data Graph Construction Ablation

<!-- BEGIN GENERATED table=phase3_conclusion run_id=ablations-9faf2a74 -->
Ablations on Dataset B (sigma 0.1, M=16, efC=100) over a 5x5 data-seed by build-seed matrix show that the recall gap came from the original construction defaults: each new node selected M0 neighbors and back-filled candidates rejected by the neighbor-selection heuristic (new_node_degree="M0", keep_pruned=true). Switching to reference-style semantics (new_node_degree="M", keep_pruned=false) raised mean recall@10 at ef=200 from 0.9972 to 0.9998, reduced the mean number of nodes that fail 1-NN self-retrieval from 166.16 to 9.32, and reduced Dataset B build time from 8.51 s to 0.93 s within the same ablation run.

Protecting each node's last incoming edge reduced BFS-unreachable nodes to 0 without improving recall (0.9726), so pure in-edge starvation does not explain the gap. Leading hypothesis, not isolated by a direct test: back-filled intra-cluster links crowd out long-range links and reduce navigability. About 9.32 nodes per index still fail self-retrieval, so target T5 remains a MISS.
<!-- END GENERATED -->

---

## 4. Technical Q&As (12 Grounded Systems Questions)

### Q1: "How do HNSW insert and search work step-by-step?"
> "During **search**, traversal begins at the top layer $L_{max}$ at the global entry point. At each layer $l > 0$, greedy descent hops to the neighbor closest to query $q$ until no neighbor is closer ($ef=1$). That closest node becomes the entry point for layer $l - 1$. At layer 0, the algorithm switches to beam search with beam width $ef_{search}$, maintaining a min-heap candidate queue and a max-heap of the top-$k$ nearest neighbors found.
>
> During **insert**, a new vector is assigned a maximum layer $l_{new}$ drawn from an exponential distribution. The algorithm traverses from $L_{max}$ down to $l_{new} + 1$ using greedy descent to find the closest entry point. From $\min(L_{max}, l_{new})$ down to 0, it executes beam search with width $ef_{construction}$ to find candidate neighbors, applies heuristic neighbor selection to connect up to $M$ edges (or $M_0$ on layer 0), adds bidirectional edges, and shrinks overflowing neighbor lists."

### Q2: "What are the trade-offs between M, efConstruction, and efSearch?"
> - **$M$ (connectivity degree):** Controls index memory footprint ($M \times 4$ bytes per node per layer) and search speed. Higher $M$ improves recall on high-dimensional or clustered data but increases distance evaluations per hop and index build time. Typical range: 16 to 64.
> - **$ef_{construction}$ (build beam width):** Controls graph build quality. Higher $ef_c$ explores a wider candidate pool during insertion, preventing early nodes from becoming disconnected, but scales build time proportionally. Has zero impact on memory footprint.
> - **$ef_{search}$ (query beam width):** Runtime parameter controlling the recall-vs-latency trade-off. Can be tuned dynamically per query without re-indexing. Increasing $ef_{search}$ increases recall monotonically until the graph connectivity limit is reached."

### Q3: "Why use heuristic neighbor selection (Algorithm 4) instead of simply picking the closest M neighbors?"
> "Connecting only to the closest $M$ neighbors creates dense local cliques: all neighbors cluster in the same direction, leaving the node without connections to neighboring clusters. Algorithm 4 enforces directional diversity: a candidate $e$ is connected to base node $q$ only if $e$ is closer to $q$ than to any already-selected neighbor $r$ ($d(e, q) < d(e, r)$). This prunes redundant triangles, favors neighbors in diverse geometric directions, and preserves long-range navigability."

*(Continued in [Part 2](INTERVIEW_PREP_part2.md) for Q4–Q12 and Sections 5–7)*
