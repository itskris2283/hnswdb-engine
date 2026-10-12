# HNSWDB: Professional Resume Bullets

<!-- BEGIN GENERATED table=resume_bullets run_id=fashion_mnist_20261010_174828_0a9f92db -->
* Built an HNSW vector database from scratch in C++17 (AVX2/FMA kernels, metadata-filtered search, CRC32 atomic persistence, pybind11 bindings, FastAPI and CLI) with 9 C++ test suites and 78 Python tests; on Fashion-MNIST (60K x 784) it reaches 0.9951 recall@10 at ef=50 with p50 0.220 ms, in line with FAISS and hnswlib at the same parameters.
* Ran a 25-seed construction study on tightly clustered data that traced a recall gap to the original construction defaults, raising mean recall@10 from 0.9972 to 0.9998 (worst seed from 0.9885 to 0.9974) and cutting mean 1-NN self-retrieval failures from 166.16 to 9.32.
* Built a generated-docs pipeline in which every published benchmark number is rendered from raw run JSON, with checkers that fail on hand-edited tables, banned claims, and changed targets (13 targets frozen in git).

### Provenance Table

| Placeholder | Rendered Value | Raw Result File | JSON Key Path | Run ID |
| :--- | :--- | :--- | :--- | :--- |
| `n_cpp` | 9 | `results/raw/scorecard.json` | `targets[11].value` | `scorecard` |
| `n_py` | 78 | `results/raw/scorecard.json` | `targets[11].value` | `scorecard` |
| `fm_recall50` | 0.9951 | `results/raw/bench_fashion_mnist_0a9f92db.json` | `engines.hnswdb.seed_42.ef_results[1].recall_at_k` | `fashion_mnist_20261010_174828_0a9f92db` |
| `fm_p50` | 0.220 | `results/raw/bench_fashion_mnist_0a9f92db.json` | `engines.hnswdb.seed_42.ef_results[1].latency.p50` | `fashion_mnist_20261010_174828_0a9f92db` |
| `legacy_recall_mean` | 0.9972 | `results/raw/seed_matrix_dataset_b.json` | `engines.hnswdb_legacy.recall_1t_mean` | `seed-matrix-dbead162` |
| `cured_recall_mean` | 0.9998 | `results/raw/seed_matrix_dataset_b.json` | `engines.hnswdb_cured.recall_1t_mean` | `seed-matrix-dbead162` |
| `legacy_recall_min` | 0.9885 | `results/raw/seed_matrix_dataset_b.json` | `engines.hnswdb_legacy.recall_1t_min` | `seed-matrix-dbead162` |
| `cured_recall_min` | 0.9974 | `results/raw/seed_matrix_dataset_b.json` | `engines.hnswdb_cured.recall_1t_min` | `seed-matrix-dbead162` |
| `legacy_unreach_mean` | 166.16 | `results/raw/seed_matrix_dataset_b.json` | `engines.hnswdb_legacy.unreachable_mean` | `seed-matrix-dbead162` |
| `cured_unreach_mean` | 9.32 | `results/raw/seed_matrix_dataset_b.json` | `engines.hnswdb_cured.unreachable_mean` | `seed-matrix-dbead162` |
| `n_targets` | 13 | `benchmarks/targets.json` | `targets` | `round5-targets-frozen` |
<!-- END GENERATED -->
