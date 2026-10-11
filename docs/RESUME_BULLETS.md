# HNSWDB: Professional Resume Bullets

<!-- BEGIN GENERATED table=resume_bullets run_id=fashion_mnist_20261010_174828_0a9f92db -->
* Built an HNSW vector database from scratch in C++17 (AVX2/FMA kernels, metadata-filtered search, CRC32 atomic persistence, pybind11 bindings, FastAPI and CLI) with 9 C++ and 74 Python tests; on Fashion-MNIST (60K x 784) it reaches 0.9951 recall@10 at ef=50 with p50 0.220 ms, in line with FAISS and hnswlib at the same parameters.
* Ran a 25-seed ablation on tightly clustered data that traced a recall gap to the original construction defaults, raising mean recall@10 from 0.9972 to 0.9998 and cutting Dataset B build time from 8.51 s to 0.93 s in the same run.
* Built a generated-docs pipeline in which every published benchmark number is rendered from raw run JSON, with checkers that fail on hand-edited tables, banned claims, and changed targets (13 targets frozen in git).

### Provenance Table

| Placeholder | Rendered Value | Raw Result File | JSON Key Path | Run ID |
| :--- | :--- | :--- | :--- | :--- |
| `n_cpp` | 9 | `results/raw/scorecard.json` | `targets[11].value` | `scorecard` |
| `n_py` | 74 | `results/raw/scorecard.json` | `targets[11].value` | `scorecard` |
| `fm_recall50` | 0.9951 | `results/raw/bench_fashion_mnist_0a9f92db.json` | `engines.hnswdb.seed_42.ef_results[1].recall_at_k` | `fashion_mnist_20261010_174828_0a9f92db` |
| `fm_p50` | 0.220 | `results/raw/bench_fashion_mnist_0a9f92db.json` | `engines.hnswdb.seed_42.ef_results[1].latency.p50` | `fashion_mnist_20261010_174828_0a9f92db` |
| `legacy_recall_mean` | 0.9972 | `results/raw/seed_matrix_dataset_b.json` | `engines.hnswdb_legacy.recall_1t_mean` | `seed-matrix-dbead162` |
| `cured_recall_mean` | 0.9998 | `results/raw/seed_matrix_dataset_b.json` | `engines.hnswdb_cured.recall_1t_mean` | `seed-matrix-dbead162` |
| `legacy_build_s` | 8.51 | `results/raw/ablations_dataset_b.json` | `configs[degree_M0__keep_True__protect_False].dataset_b.build_time_s` | `ablations-9faf2a74` |
| `cured_build_s` | 0.93 | `results/raw/ablations_dataset_b.json` | `configs[degree_M__keep_False__protect_False].dataset_b.build_time_s` | `ablations-9faf2a74` |
| `n_targets` | 13 | `benchmarks/targets.json` | `targets` | `round5-targets-frozen` |
<!-- END GENERATED -->
