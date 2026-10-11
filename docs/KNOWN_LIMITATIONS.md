# Known Limitations & Performance Characteristics

This document transparently records verified engineering limitations, dataset sensitivities, and trade-offs in HNSWDB.

---

## 1. Clustered Distribution Weakness (Implementation-Specific Construction Weakness)

- **Symptom:** On synthetic distributions with tight, well-separated Gaussian clusters (50 clusters, standard deviation $\sigma=0.1$ across 20,000 vectors in 64 dimensions, Dataset B), legacy HNSW graph construction with standard $ef_{construction}=100$ reached an early recall ceiling at $ef_{search}=200$ (see generated tables in [INVESTIGATIONS.md](INVESTIGATIONS.md)).
- **Isolation & Resolution:** In Phase 3 ablations (`results/raw/ablations_study.json`), the issue was isolated to initial layer-0 degree over-allocation (`new_node_degree="M0"`, 32 links) combined with retaining pruned candidates (`keep_pruned=true`), which crowded out inter-cluster bridges. Setting defaults to `new_node_degree="M"` (16 links) and `keep_pruned=false` cured the recall gap, achieving 1.0000 recall at $ef=200$.
- **Residual Limitation:** Unreachable nodes are reduced from over five hundred to single digits, but non-zero on clustered distributions (see generated seed matrix and ablation tables in [INVESTIGATIONS.md](INVESTIGATIONS.md)), causing target T5 to be reported as a MISS.

---

## 2. Build-time comparison with hnswlib and FAISS

Earlier builds were several times slower than hnswlib on small datasets. Same-run ablations show that most of that gap came from the original construction defaults (M0 neighbors for new nodes and back-filled candidates). With the current defaults, build times are comparable to hnswlib and FAISS in the generated tables. Absolute timings on this laptop vary by two to three times between sessions, so only within-run ratios are reported. No profiler was available on the host, so remaining differences are not attributed to a specific cause.

---

## 3. Tombstone Garbage Collection & Compaction

- **Behavior:** Deleted vectors are tombstoned via bit flags in `deleted_` (`uint8_t` array) and excluded from query results in $O(1)$.
- **Limitation:** Tombstoned nodes retain their graph link positions until explicit `index.compact()` is executed. In workloads with high churn, calling `compact()` rebuilds a clean graph to reclaim memory and maintain search recall.

---

## 4. Platform and Sanitizer Verification

- **Host Platform:** Development and validation were performed on Windows 11 (MSVC 2022 Release /O2, AVX2+FMA SIMD kernel).
- **Sanitizers:** Sanitizers were not executed locally on Windows MSVC. A CI workflow for Linux (GCC ASan/UBSan, Clang TSan) is provided in `.github/workflows/ci.yml`.

---

## 5. Upstream Dataset Availability (SIFT1M)

- **Observation:** Direct automated download of `sift-128-euclidean.hdf5` from `http://ann-benchmarks.com` fails with HTTP 403 Forbidden.
- **Status:** Per Rule 0, SIFT1M is not evaluated in local benchmarks, and all benchmark claims rely strictly on locally verified datasets: Fashion-MNIST (60K $\times$ 784d), Dataset B ($\sigma=0.1$, $\sigma=0.3$, $\sigma=1.0$), and Dataset A (5K $\times$ 32d).

