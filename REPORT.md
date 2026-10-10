# HNSWDB: Final Engineering & Verification Report

## 1. Executive Summary & Audited Scorecard

**HNSWDB** is an approximate nearest neighbor (ANN) vector database implemented from first principles in C++17 with Python bindings (`pybind11`), FastAPI service, Typer CLI, and automated test and benchmark suites.

Every measured number in this report is programmatically generated from raw JSON artifacts in `results/raw/`. No metrics are hand-typed.

### Audited Scorecard (Fixed Targets Evaluated by Code)

<!-- BEGIN GENERATED table=scorecard run_id=A_20261010_100807_7d8938d0 -->
**Dataset**: A (5000 x 32d) | **M**=16 | **efC**=100 | **k**=10

| ID | Target Description | Value | Status |
| :--- | :--- | :--- | :--- |
| `T1` | Recall@10 >= 0.99 on Dataset A (M=16, efC=100, ef=50) | `0.9987` | **PASS** |
| `T2` | Recall@10 >= 0.97 on Dataset B sigma=0.1 (M=16, efC=100, ef=200) | `0.9716` | **PASS** |
| `T3` | Recall@10 gap vs hnswlib <= 0.02 on Dataset B sigma=0.1 (ef=200) | `0.0284` | **MISS** |
| `T4` | Approximation ratio <= 1.05 on Dataset A (ef=50) | `1.0000` | **PASS** |
| `T5` | Build time within 5x of hnswlib on Dataset A | `3.0306` | **PASS** |
| `T6` | Single-thread QPS >= 5000 on Dataset A (ef=50) | `36574.0368` | **PASS** |
| `T7` | Recall monotonically non-decreasing with ef on Dataset A | `True` | **PASS** |
| `T8` | All C++ tests pass (CTest) | `9/9 passed` | **PASS** |
| `T9` | All Python tests pass (pytest) | `45/45 passed` | **PASS** |
| `T10` | Provenance check passes | `verified` | **PASS** |
<!-- END GENERATED -->

*Note on T3 (MISS): On tightly-clustered synthetic data ($\sigma=0.1$), HNSWDB exhibits a recall gap vs hnswlib at $ef_c=100$ that exceeds the target threshold. This limitation is investigated and documented in Section 2 and [INVESTIGATIONS.md](docs/INVESTIGATIONS.md).*

---

## 2. Corrections & Retractions from Earlier Rounds

In the spirit of complete engineering honesty, this section explicitly documents all claims from earlier rounds that were found to be erroneous or unverified upon audit:

| Earlier Claim / Metric | Error Identified | Technical Analysis | Correction |
| :--- | :--- | :--- | :--- |
| **Spliced Benchmark Tables** | BENCHMARKS.md mixed recall from one benchmark run with latency/QPS from a different run. | Lack of single-run pipeline automation. | Eliminated all hand-typed tables. All tables are now rendered by `scripts/render_docs.py` from a single `run_id`. |
| **"3,080,326 duplicate links"** | Claimed duplicate link count in a 5,000-node index. | Layer 0 holds at most $5,000 \times 32 = 160,000$ links; count was impossible. | Retracted. Verified that snapshot graph invariants have exactly 0 duplicate links. |
| **Dataset B Recall Gap** | Claimed Dataset B was an extreme distribution and efC=400 was sufficient, ignoring that FAISS and hnswlib achieve ~1.0 at efC=100. | Incomplete investigation of construction algorithm. | Investigated in depth: 505 nodes become search-unreachable at efC=100 due to heuristic pruning during overflow reselection. Documented in [INVESTIGATIONS.md](docs/INVESTIGATIONS.md) and [KNOWN_LIMITATIONS.md](docs/KNOWN_LIMITATIONS.md). |
| **Approximation Ratio = 31.9** | Fashion-MNIST approximation ratio was reported as ~31.9 instead of ~1.0. | `fashion-mnist-784-euclidean.hdf5` stores Euclidean distances ($\approx 482$), while `approx_ratio` assumed squared distances and took a second square root ($\sqrt{482} \approx 22$). | Fixed distance semantics in `benchmarks/datasets_synth.py`. Approximation ratio is verified to be 1.000. |
| **Permissive Whitelist** | `docs/claims_whitelist.txt` whitelisted specific benchmark numbers, hiding hardcoded claims from the checker. | Checker design allowed numbers instead of enforcing generated provenance. | Deleted `claims_whitelist.txt`. Replaced with strict `docs/claims_allowlist.txt` containing only architectural constants and parameter bounds. |
| **Linux Sanitizer Claims** | Earlier reports claimed clean ThreadSanitizer status despite testing solely on Windows (MSVC). | MSVC does not support TSan. | Clarified host environment: local Windows 11 host with MSVC; CI configuration provided for Linux ASan/UBSan/TSan. |

---

## 3. Verified Benchmark Results

### 3.1 Fashion-MNIST ($60,000 \times 784\text{d}, M=16, ef_c=100, k=10$)

<!-- BEGIN GENERATED table=full_fashion_mnist run_id=fashion_mnist_20261010_113121_90c0d1d3 -->
**Dataset**: fashion_mnist (60000 x 784d) | **M**=16 | **efC**=100 | **k**=10

| Engine | ef_search | Recall@10 | Approx Ratio | p50 (ms) | Sequential QPS |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **hnswdb** | 10 | 0.9705 | 1.0009 | 0.204 | 4,615 |
| **hnswdb** | 50 | 0.9978 | 1.0001 | 0.452 | 2,085 |
| **hnswdb** | 200 | 0.9996 | 1.0000 | 1.107 | 858 |
| **faiss** | 10 | 0.9264 | 1.0009 | 0.149 | 6,235 |
| **faiss** | 50 | 0.9948 | 1.0001 | 0.365 | 2,588 |
| **faiss** | 200 | 0.9990 | 1.0000 | 1.623 | 492 |
| **hnswlib** | 10 | 0.9277 | 1.0009 | 0.291 | 2,816 |
| **hnswlib** | 50 | 0.9951 | 1.0001 | 0.813 | 1,033 |
| **hnswlib** | 200 | 0.9991 | 1.0000 | 1.941 | 451 |
<!-- END GENERATED -->

### 3.2 Dataset B Clustered Synthetic ($20,000 \times 64\text{d}, M=16, ef_c=100, k=10$)

<!-- BEGIN GENERATED table=full_dataset_b run_id=B_s01_20261010_100918_0d563c34 -->
**Dataset**: B_s01 (20000 x 64d) | **M**=16 | **efC**=100 | **k**=10

| Engine | ef_search | Recall@10 | Approx Ratio | p50 (ms) | Sequential QPS |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **hnswdb** | 10 | 0.9101 | 1.0735 | 0.021 | 43,684 |
| **hnswdb** | 50 | 0.9711 | 1.0017 | 0.039 | 23,262 |
| **hnswdb** | 200 | 0.9716 | 1.0017 | 0.085 | 11,346 |
| **faiss** | 10 | 0.8508 | 1.1503 | 0.035 | 24,424 |
| **faiss** | 50 | 0.9943 | 1.0336 | 0.055 | 17,057 |
| **faiss** | 200 | 0.9994 | 1.0000 | 0.159 | 5,911 |
| **hnswlib** | 10 | 0.8513 | 1.0253 | 0.028 | 31,381 |
| **hnswlib** | 50 | 0.9979 | 1.0001 | 0.057 | 15,685 |
| **hnswlib** | 200 | 1.0000 | 1.0000 | 0.089 | 10,755 |
<!-- END GENERATED -->

---

## 4. Test Suite Summary

- **C++ GoogleTest Suite:** 9 / 9 tests passing (CTest)
- **Python Pytest Suite:** 45 / 45 tests passing (`pytest tests/python/`)
- **Strict Claims Check:** PASSED (`python scripts/check_claims.py`)
- **Documentation Link Check:** PASSED (`python scripts/check_docs_links.py`)
- **Provenance Verification:** PASSED (`python scripts/check_provenance.py`)

---

## 5. Reproduction Instructions

To reproduce all benchmarks, tests, and documentation rendering from scratch:

```bash
# 1. Install dependencies
pip install -e ".[dev]"

# 2. Run C++ tests
cd build/rel && ctest --output-on-failure -C Release && cd ../..

# 3. Run Python tests
python -m pytest tests/python/ -q

# 4. Run Benchmark Pipeline (generates raw JSON results)
python scripts/run_pipeline.py --config smoke --datasets A --seeds 42

# 5. Generate Scorecard
python scripts/make_scorecard.py

# 6. Render Documentation
python scripts/render_docs.py

# 7. Verify Claims and Provenance
python scripts/check_claims.py
python scripts/check_provenance.py
python scripts/check_docs_links.py
```
