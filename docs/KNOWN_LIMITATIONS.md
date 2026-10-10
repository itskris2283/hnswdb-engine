# Known Limitations & Performance Characteristics

This document transparently records verified engineering limitations, dataset sensitivities, and trade-offs in HNSWDB.

---

## 1. Clustered Distribution Weakness (Implementation-Specific Construction Weakness)

- **Symptom:** On synthetic distributions with tight, well-separated Gaussian clusters (50 clusters, standard deviation $\sigma=0.1$ across 20,000 vectors in 64 dimensions, Dataset B), HNSW graph construction with standard $ef_{construction}=100$ reaches an early recall ceiling at $ef_{search}=200$ (see generated tables in [INVESTIGATIONS.md](INVESTIGATIONS.md)).
- **Baseline Context:** This is an implementation-specific construction weakness in HNSWDB; reference baselines (`hnswlib` and `FAISS`) do not exhibit this problem on the identical dataset, reaching higher recall at the same parameters.
- **Leading Hypotheses (Not Yet Isolated):**
  - *Hypothesis A (Neighbor Degree Pressure):* In HNSWDB, new nodes select up to $M_0=32$ neighbors at layer 0 upon insertion, whereas hnswlib selects $M=16$ neighbors for a new element at all layers (allowing existing neighbors to expand up to $M_0$). Selecting 32 links immediately may double reverse-link pruning pressure on existing nodes.
  - *Hypothesis B (Discarded Candidate Back-fill):* HNSWDB's heuristic selection retains discarded candidates to back-fill up to $M$ connections (`keepPrunedConnections`), whereas hnswlib's `getNeighborsByHeuristic2` does not back-fill when candidates exceed $M$.
  - *Hypothesis C (Unprotected In-Edges):* When neighbor lists overflow, links to early-inserted nodes may be evicted without verifying if the evicted node retains at least one incoming edge, causing nodes to become search-unreachable.
- **Workarounds:**
  - Increasing $ef_{construction}$ to 400 provides a broader candidate pool during build, restoring full recall on the exact same dataset (see [INVESTIGATIONS.md](INVESTIGATIONS.md)).
  - On continuous manifold datasets (such as Fashion-MNIST $784$-d), standard $ef_{construction}=100$ achieves target recall at $ef_{search}=200$ (see generated tables in [BENCHMARKS.md](BENCHMARKS.md)).

---

## 2. Build-Time Gap vs hnswlib/FAISS: Causes Not Yet Isolated

- **Observation:** HNSWDB single-thread index construction is slower than `hnswlib` and `FAISS`.
- **Status:** Causes not yet isolated. Early speculations regarding cache lines and function pointers have been removed until Phase 5 empirical work-counting and profiling provide concrete evidence.

---

## 3. Tombstone Garbage Collection & Compaction

- **Behavior:** Deleted vectors are tombstoned via bit flags in `deleted_` (`uint8_t` array) and excluded from query results in $O(1)$.
- **Limitation:** Tombstoned nodes retain their graph link positions until explicit `index.compact()` is executed. In workloads with high churn, calling `compact()` rebuilds a clean graph to reclaim memory and maintain search recall.

---

## 4. Platform and Sanitizer Verification

- **Host Platform:** Development and validation were performed on Windows 11 (MSVC 2022 Release /O2, AVX2+FMA SIMD kernel).
- **Sanitizers:** Sanitizers were not executed locally on Windows MSVC. A CI workflow for Linux (GCC ASan/UBSan, Clang TSan) is provided in `.github/workflows/ci.yml`.
