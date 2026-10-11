# HNSWDB: Final Engineering & Verification Report

## 1. Executive Summary & Audited Scorecard

**HNSWDB** is an approximate nearest neighbor (ANN) vector database implemented from first principles in C++17 with Python bindings (`pybind11`), FastAPI service, Typer CLI, and automated test and benchmark suites.

Every measured number in this report is programmatically generated from raw JSON artifacts in `results/raw/`. No metrics are hand-typed.

### Audited Scorecard (Fixed Targets Evaluated by Code)

<!-- BEGIN GENERATED table=scorecard run_id=A_20261010_174754_a9f1a6f6 -->
**Scorecard (multiple datasets)**

| ID | Target Description | Value | Status | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `T1` | Fashion-MNIST (M=16, efC=100): recall@10 >= 0.99 at some ef <= 100 and >= 0.999 at some ef <= 400 | `0.9992` | **PASS** |  |
| `T2` | Dataset B (sigma=0.1, efC=100): recall@10 at ef=200, mean over 5 data seeds x 5 build seeds, >= 0.999 | `0.9998` | **PASS** |  |
| `T3` | Dataset B: mean recall gap vs hnswlib (same seeds) <= 0.002 | `0.0005` | **PASS** |  |
| `T4` | Corrected self-retrieval rate >= 0.999 on Dataset B and on Fashion-MNIST | `0.9976` | **MISS** | Dataset B: 0.9995, Fashion-MNIST: 0.9976. Baselines under the same metric: hnswlib (0.9994 / 0.9978), FAISS (0.9989 / 0.9952) |
| `T5` | Unreachable nodes == 0 (Dataset B mean, Fashion-MNIST) | `12.0000` | **MISS** | Dataset B mean: 9.32, Fashion-MNIST (in sample): 12. Baselines under the same metric: hnswlib (3.00 / 11), FAISS (5.40 / 24) |
| `T6` | HNSWDB single-thread vs multi-thread build recall difference < 0.003 | `0.0014` | **PASS** |  |
| `T7` | Sequential QPS at recall@10 = 0.99 within 3x of hnswlib (Fashion-MNIST, equal-recall interpolation) | `1.1000` | **PASS** |  |
| `T8` | Build time, 1 thread, within 3x of hnswlib on Fashion-MNIST and Dataset B | `0.6200` | **PASS** |  |
| `T9` | Build time at max physical threads within 3x of hnswlib (MISS is acceptable if reported) | `1.3900` | **PASS** |  |
| `T10` | RSS delta within 15% of hnswlib on Fashion-MNIST | `0.0335` | **PASS** |  |
| `T11` | Approximation ratio of exact search == 1.0 +/- 1e-6; Fashion-MNIST ratio <= 1.05 | `1.0000` | **PASS** |  |
| `T12` | All C++ and Python tests pass (counts from run output) | `9 C++ / 72 Python passed` | **PASS** |  |
| `T13` | Sanitizers: zero findings (status NOT RUN if not executed) | `NOT RUN (Windows host)` | **NOT RUN** |  |

*Target T5 uses the self-retrieval unreachable metric.*
*Baselines under the same metric: hnswlib (Dataset B: 3.00 unreachable / 0.9994 rate; Fashion-MNIST: 11 unreachable / 0.9978 rate); FAISS (Dataset B: 5.40 unreachable / 0.9989 rate; Fashion-MNIST: 24 unreachable / 0.9952 rate).*
<!-- END GENERATED -->

*Note on T5 (MISS): On tightly-clustered synthetic data ($\sigma=0.1$), HNSWDB exhibits 9.32 mean unreachable nodes (target is 0). This limitation is investigated and documented in Section 2 and [INVESTIGATIONS.md](docs/INVESTIGATIONS.md).*

### Phase 3 Conclusion: Clustered-Data Gap Analysis

<!-- BEGIN GENERATED table=phase3_conclusion run_id=ablations-9faf2a74 -->
Ablations on Dataset B (sigma 0.1, M=16, efC=100) over a 5x5 data-seed by build-seed matrix show that the recall gap came from the original construction defaults: each new node selected M0 neighbors and back-filled candidates rejected by the neighbor-selection heuristic (new_node_degree="M0", keep_pruned=true). Switching to reference-style semantics (new_node_degree="M", keep_pruned=false) raised mean recall@10 at ef=200 from 0.9972 to 0.9998, reduced the mean number of nodes that fail 1-NN self-retrieval from 166.16 to 9.32, and reduced Dataset B build time from 8.51 s to 0.93 s within the same ablation run.

Protecting each node's last incoming edge reduced BFS-unreachable nodes to 0 without improving recall (0.9726), so pure in-edge starvation does not explain the gap. Leading hypothesis, not isolated by a direct test: back-filled intra-cluster links crowd out long-range links and reduce navigability. About 9.32 nodes per index still fail self-retrieval, so target T5 remains a MISS.
<!-- END GENERATED -->

---

## 2. Corrections & Retractions from Earlier Rounds

In the spirit of complete engineering honesty, this section explicitly documents all claims and framing from earlier rounds that were found to be erroneous, fabricated, or unverified upon audit:

<!-- BEGIN CORRECTIONS -->
| Earlier Claim / Metric | Round / Context | Error Identified | Correction & Evidence |
| :--- | :--- | :--- | :--- |
| **"3,080,326 duplicate links"** | Round 2 / docs/INVESTIGATIONS.md | Claimed duplicate link count in a 5,000-node index ($M=16, M_0=32$). | Retracted as mathematically impossible: layer 0 holds at most $5,000 \times 32 = 160,000$ links total. Invariant tests confirm 0 duplicate links exist. |
| **"Recall 0.9912 -> 0.9994 from dedup fix"** | Round 2 / docs/INVESTIGATIONS.md | Attributed a recall jump from 0.9912 to 0.9994 to deduplication logic. | Retracted: the deduplication change had no measurable effect on recall (0.0000 delta across 5 seeds). Ablation study showed difference was noise. |
| **Dedup fix effect** | Round 3 / docs/INVESTIGATIONS.md | Claimed dedup resolved clustered-data recall plateau. | Retracted: the duplicate-links change did not alter connectivity; the underlying issue remained unaddressed until the Phase 3 ablation study. |
| **"Matches and surpasses baselines"** | Round 2-3 / README.md, REPORT.md | Headline claimed HNSWDB matches and surpasses FAISS and hnswlib. | Retracted: HNSWDB achieves competitive recall and latency parity on Fashion-MNIST, but loses on single-thread build time (1.05x-2.3x) and on tightly clustered data. |
| **CI and Sanitizer PASS claims** | Round 2-3 / REPORT.md, README.md | Claimed clean ThreadSanitizer (TSan) status and passing CI. | Retracted: MSVC on Windows does not support TSan. Local host did not run Linux sanitizers. Workflow defined in `.github/workflows/ci.yml`. Status is honest `NOT RUN (Windows host)`. |
| **FAISS template specialization attribution** | Round 2-3 / docs/TRADEOFFS.md | Attributed FAISS's build-time advantage to `L2SqrSIMD16Ext` template specialization. | Retracted: L2SqrSIMD16Ext is an hnswlib function, not FAISS's. Template specialization was never shown to explain any gap. Same-run ablations show most of the earlier build-time gap came from the original construction defaults. |
| **Memory layout and function-pointer build-gap claims** | Round 3 / docs/TRADEOFFS.md, KNOWN_LIMITATIONS.md | Speculated that contiguous vs array-of-vectors layout or indirect function calls caused the build-time gap. | Retracted: unverified speculation. No profiler was available on the host; the gap was explained by same-run ablations and distance-computation counts, not instruction profiling. |
| **"Real algorithmic limitation" framing** | Round 4 / docs/INVESTIGATIONS.md | Framed the clustered-data recall gap as an inherent algorithmic limitation of HNSW. | Retracted: hnswlib and FAISS reach the target on identical data. The gap came from HNSWDB's original construction defaults (see the generated conclusion in INVESTIGATIONS.md). |
| **"505 unreachable nodes" and "reduced unreachable nodes to 2"** | Round 4 / docs/INVESTIGATIONS.md | Quoted single-seed 505 and misattributed BFS count 2 as general reachability improvement. | Corrected: 505 was the self-retrieval count for one data seed; the 25-seed means are given in the generated conclusion. The value 2 came from the BFS metric on a different row and was misattributed. |
| **Resume-bullet figures (earlier generator)** | Round 4 / docs/RESUME_BULLETS.md | Used 94.8% distance share contradicting work counting, and legacy recall as result. | Corrected: an earlier generated bullet used a distance-share figure of 94.8% that contradicts the work-counting table, and another presented legacy recall as a result. The generator was rewritten (C4). |
| **Round-4 loosening of fixed targets** | Round 4 / REPORT.md | Loosened fixed targets to force PASS results (relaxed recall 0.999 to 0.97, gap 0.002 to 0.02, build ratio 3x to 5x). | Retracted: all targets frozen byte-for-byte in `benchmarks/targets.json` and tagged `round5-targets-frozen`. Evaluated strictly by code with zero target loosening. |
| **Spliced Benchmark Tables** | Round 3 / docs/BENCHMARKS.md | Spliced recall from one run with latency/QPS from a different run. | Retracted: hand-typed tables eliminated. All documentation blocks rendered strictly from a single `run_id` by `scripts/render_docs.py`. |
<!-- END CORRECTIONS -->

---

## 3. Verified Benchmark Results

### 3.1 Fashion-MNIST ($60,000 \times 784\text{d}, M=16, ef_c=100, k=10$)

<!-- BEGIN GENERATED table=full_fashion_mnist run_id=fashion_mnist_20261010_174828_0a9f92db -->
**Dataset**: fashion_mnist (60000 x 784d) | **M**=16 | **efC**=100 | **k**=10

| Engine | ef_search | Recall@10 | Approx Ratio | p50 (ms) | Sequential QPS |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **hnswdb** | 10 | 0.9278 | 1.0034 | 0.103 | 9,071 |
| **hnswdb** | 50 | 0.9951 | 1.0003 | 0.220 | 4,259 |
| **hnswdb** | 200 | 0.9992 | 1.0000 | 0.478 | 2,048 |
| **faiss** | 10 | 0.9264 | 1.0032 | 0.110 | 8,357 |
| **faiss** | 50 | 0.9948 | 1.0003 | 0.230 | 4,125 |
| **faiss** | 200 | 0.9990 | 1.0001 | 0.614 | 1,519 |
| **hnswlib** | 10 | 0.9277 | 1.0030 | 0.111 | 8,577 |
| **hnswlib** | 50 | 0.9951 | 1.0002 | 0.296 | 3,118 |
| **hnswlib** | 200 | 0.9991 | 1.0000 | 0.706 | 1,346 |
<!-- END GENERATED -->

### 3.2 Dataset B Clustered Synthetic ($20,000 \times 64\text{d}, M=16, ef_c=100, k=10$)

<!-- BEGIN GENERATED table=full_dataset_b run_id=B_s01_20261010_174800_8a2302be -->
**Dataset**: B_s01 (20000 x 64d) | **M**=16 | **efC**=100 | **k**=10

| Engine | ef_search | Recall@10 | Approx Ratio | p50 (ms) | Sequential QPS |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **hnswdb** | 10 | 0.8461 | 1.0539 | 0.018 | 55,211 |
| **hnswdb** | 50 | 0.9978 | 1.0001 | 0.069 | 14,263 |
| **hnswdb** | 200 | 1.0000 | 1.0000 | 0.090 | 11,029 |
| **faiss** | 10 | 0.8508 | 1.1503 | 0.084 | 2,905 |
| **faiss** | 50 | 0.9943 | 1.0336 | 0.290 | 2,186 |
| **faiss** | 200 | 0.9994 | 1.0000 | 0.387 | 1,571 |
| **hnswlib** | 10 | 0.8513 | 1.0253 | 0.030 | 27,869 |
| **hnswlib** | 50 | 0.9979 | 1.0001 | 0.056 | 17,846 |
| **hnswlib** | 200 | 1.0000 | 1.0000 | 0.106 | 9,258 |
<!-- END GENERATED -->

---

## 4. Test Suite Summary

- **C++ GoogleTest Suite:** 9 / 9 test suites passing (`ctest`)
- **Python Pytest Suite:** 72 / 72 tests passing (`python -m pytest tests/python -v`)
- **Strict Claims Check:** PASSED (`python scripts/check_claims.py`)
- **Targets Frozen Check:** PASSED (`python scripts/check_targets_unchanged.py`)
- **Documentation Link Check:** PASSED (`python scripts/check_docs_links.py`)
- **Provenance Verification:** PASSED (`python scripts/check_provenance.py`)
- **README Live Verification:** PASSED (`python scripts/verify_readme.py`)

---

## 5. What Is Not Verified

In compliance with empirical verification standards, the following areas remain unverified in this execution environment:

1. **Linux Dynamic Sanitizers (ASan, UBSan, TSan):**
   - The primary benchmarking and test environment is Windows 11 with MSVC 2022. MSVC does not support ThreadSanitizer (TSan), and local WSL lacks an installed Linux distribution.
   - Containerized verification scripts are provided at [Dockerfile.linux-checks](docker/Dockerfile.linux-checks) and [run_linux_checks.sh](scripts/run_linux_checks.sh), and CI workflows are defined in [.github/workflows/ci.yml](.github/workflows/ci.yml). Local status remains honestly reported as `NOT RUN (Windows host)`.

2. **SIFT1M Real-World Benchmark:**
   - The INRIA FTP mirror for `sift1m.tar.gz` returns HTTP 403 Forbidden. SIFT1M metrics were omitted rather than fabricated or reconstructed from unverified sources. See [KNOWN_LIMITATIONS.md](docs/KNOWN_LIMITATIONS.md).

3. **Linux `perf` Hardware PMU Profiling:**
   - Linux hardware PMU counters (`perf stat`, cache miss counters) were not collected on the Windows host. High-resolution synthetic work counting scripts were executed instead to measure exact distance instruction and work fractions.

---

## 6. Reproduction Instructions

To reproduce all benchmarks, test suites, and rendered documentation from scratch:

```bash
# 1. Install dependencies in editable mode
python -m pip install -e ".[dev]"

# 2. Run C++ GoogleTest suite
ctest --test-dir build/rel -C Release --output-on-failure

# 3. Run Python PyTest suite
python -m pytest tests/python -v

# 4. Run Full Benchmark Pipeline across datasets (Fashion-MNIST, Dataset B, Dataset A)
python scripts/run_pipeline.py --config full --datasets fashion_mnist,B,A --threads 1,2,4,8

# 5. Run Auxiliary Benchmarks (Multithread build, Filtered search, Work counting)
python benchmarks/run_multithread_build.py
python benchmarks/run_filtered_search.py
python benchmarks/run_work_counting.py

# 6. Generate Audited Scorecard
python scripts/make_scorecard.py

# 7. Render All Markdown Tables from Raw JSON
python scripts/render_docs.py

# 8. Run Complete Verification and Provenance Checkers
python scripts/check_targets_unchanged.py
python scripts/check_claims.py
python scripts/check_provenance.py
python scripts/check_docs_links.py
python scripts/verify_readme.py
```
