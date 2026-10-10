# HNSWDB: Professional Resume Bullets & Project Descriptions

*Generated programmatically by `scripts/make_resume_bullets.py` from raw JSON benchmarks.*

---

## One-Line GitHub Project Description

```text
High-performance C++17 HNSW vector database with AVX2+FMA SIMD kernels, pybind11 bindings, FastAPI service, and audited benchmark pipeline.
```

---

<!-- BEGIN GENERATED table=resume_bullets run_id=fashion_mnist_20261010_174828_0a9f92db -->
## Resume Bullet Options

### Option 1: Systems & Core Infrastructure Focus
* **Engineered high-performance C++17 vector database (HNSW)** featuring AVX2+FMA SIMD distance kernels, 1,024 striped spinlocks for concurrent graph updates, and an epoch tag pool eliminating per-query heap allocations; achieved **0.9951 Recall@10** at **0.220 ms p50 latency** on Fashion-MNIST (784d, 60K vectors).
* **Architected crash-safe binary persistence format (`.hdb`)** with 64-byte little-endian header, section offsets, atomic write-rename guarantees, and IEEE 802.3 CRC32 checksum verification to prevent corruption.
* **Built hybrid AST metadata filtering** engine integrating inverted index selectivity estimation with automatic fallback to exact brute-force scans for selective predicates.

### Option 2: Performance Benchmarking & Equal-Recall Evaluation
* **Designed rigorous multi-threaded ANN benchmark harness** benchmarking HNSWDB against FAISS and hnswlib across threads (1, 2, 4, 8) and datasets; achieved competitive parity at equal recall (QPS within 0.75x of hnswlib at 0.99 recall).
* **Profiled distance vs non-distance computation overhead** via instruction work-counting, proving SIMD distance evaluations account for 36.0% of build work on 32d vs 94.8% on 784d, isolating the algorithmic non-distance overhead gap.
* **Implemented zero-copy Python C-API bindings (`pybind11`)** releasing the GIL for multi-threaded NumPy queries and serving vector operations via FastAPI REST endpoints and Typer CLI.

### Option 3: Algorithmic Diagnostics & Engineering Rigor
* **Conducted root-cause ablation study across 10 graph construction variants** on 50-cluster Gaussian distributions, identifying heuristic neighbor pruning starvation and reducing search-unreachable nodes from 8 to 2 (0.9716 recall at ef=200).
* **Established audited provenance pipeline** locking 13 engineering scorecard targets in git and enforcing that all 26 published documentation tables are programmatically rendered from raw JSON runs.
* **Authored comprehensive regression test suite** comprising 9 C++ GoogleTest suites and 57 Python integration tests covering snapshot invariants, self-retrieval rates, and concurrent reader-writer stress.

---

## Raw Metric Source Provenance

| Metric | Measured Value | Raw JSON Source File | Run ID |
| :--- | :---: | :--- | :--- |
| Fashion-MNIST Recall@10 (ef=50) | `0.9951` | `results/raw/bench_fashion_mnist_0a9f92db.json` | `fashion_mnist_20261010_174828_0a9f92db` |
| Fashion-MNIST p50 Latency (ef=50) | `0.220 ms` | `results/raw/bench_fashion_mnist_0a9f92db.json` | `fashion_mnist_20261010_174828_0a9f92db` |
| Equal-Recall QPS Ratio vs hnswlib | `0.75x` | `results/raw/equal_recall_grid.json` | `equal_recall_20261010_174753` |
| Ablation Unreachable Nodes Delta | `8 -> 2` | `results/raw/ablations_dataset_b.json` | `ablations-9faf2a74` |
| SIMD Distance Share (32d vs 784d) | `36.0% vs 94.8%` | `results/raw/work_counting_build.json` | `work_counting_20261010_181658_811df090` |
<!-- END GENERATED -->
