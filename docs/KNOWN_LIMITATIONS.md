# Known Limitations & Performance Characteristics

This document transparently records verified engineering limitations, dataset sensitivities, and trade-offs in HNSWDB.

---

## 1. Clustered-data weakness (reduced, not eliminated)

Symptom: On tightly clustered synthetic data (Dataset B: 50 clusters, sigma 0.1, 20,000 vectors, 64 dimensions), the original construction defaults gave lower and more variable recall than hnswlib and FAISS at equal parameters (see the generated seed-matrix table in INVESTIGATIONS.md).

Finding: Same-session ablations (results/raw/ablations_dataset_b.json) point to the original defaults (M0 neighbors for new nodes with back-filled candidates). The current defaults (M neighbors, no back-fill) follow the reference algorithm. Why back-filling hurts is a hypothesis (loss of navigability); it was not isolated by a direct test.

Residual limitation: A small number of nodes still fail 1-NN self-retrieval on clustered data (generated counts in INVESTIGATIONS.md), so target T5 is a MISS. Baseline values under the same metric are shown in the same document.

---

## 2. Build-time comparison with hnswlib and FAISS

Earlier builds were several times slower than hnswlib on small datasets. Same-run ablations show that most of that gap came from the original construction defaults (M0 neighbors for new nodes and back-filled candidates). With the current defaults, build times are comparable to hnswlib and FAISS in the generated tables. Absolute timings on this laptop vary by two to three times between sessions, so only within-run ratios are reported. No profiler was available on the host, so remaining differences are not attributed to a specific cause.

---

## 3. Tombstone Garbage Collection & Compaction

- **Behavior:** Deleted vectors are tombstoned via bit flags in `deleted_` (`uint8_t` array) and excluded from query results in $O(1)$.
- **Limitation:** Tombstoned nodes retain their graph link positions until explicit `index.compact()` is executed. In workloads with high churn, calling `compact()` rebuilds a clean graph to reclaim memory and maintain search recall.

---

## 4. Platform and Sanitizer Verification

Parallel builds insert nodes in a nondeterministic order, which produces different but equally valid graphs; this is why deterministic single-threaded builds are used in the invariant tests. Freedom from data races has not been verified locally (ThreadSanitizer: NOT RUN); the Linux CI job runs it.

- **Host Platform:** Development and validation were performed on Windows 11 (MSVC 2022 Release /O2, AVX2+FMA SIMD kernel).
- **Sanitizers:** Sanitizers were not executed locally on Windows MSVC. A CI workflow for Linux (GCC ASan/UBSan, Clang TSan) is provided in `.github/workflows/ci.yml`.
- **Existing Concurrency Tests:**
  - `ConcurrencyTest.ThreadPoolBasic` (`tests/cpp/test_concurrency.cpp`)
  - `ConcurrencyTest.ConcurrentDisjointWritersAndReaders` (`tests/cpp/test_concurrency.cpp`)
  - `ConcurrencyTest.ConcurrentSearchAndDelete` (`tests/cpp/test_concurrency.cpp`)
  - `ConcurrencyTest.ConcurrentSearchDuringSave` (`tests/cpp/test_concurrency.cpp`)
  - `test_concurrent_python_readers_and_writers` (`tests/python/test_concurrency.py`)
  - `test_concurrent_batch_threading` (`tests/python/test_concurrency.py`)

---

## 5. Upstream Dataset Availability (SIFT1M)

- **Observation:** Direct automated download of `sift-128-euclidean.hdf5` from `http://ann-benchmarks.com` fails with HTTP 403 Forbidden.
- **Status:** Per Rule 0, SIFT1M is not evaluated in local benchmarks, and all benchmark claims rely strictly on locally verified datasets: Fashion-MNIST (60K $\times$ 784d), Dataset B ($\sigma=0.1$, $\sigma=0.3$, $\sigma=1.0$), and Dataset A (5K $\times$ 32d).

