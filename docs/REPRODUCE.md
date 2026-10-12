# Reproduction Commands

To reproduce all benchmarks, test suites, and rendered documentation from scratch, run the following commands in order:

```bash
# 1. Install dependencies in editable mode
python -m pip install -e ".[dev]"

# 2. Run C++ GoogleTest suite
ctest --test-dir build/rel -C Release --output-on-failure

# 3. Run Python PyTest suite
python -m pytest tests/python -v

# 4. Primary benchmarks
python scripts/run_pipeline.py --config full --datasets fashion_mnist,B,A --threads 1,2,4,8

# 5. Clustered-data study & ablations
python benchmarks/run_seed_matrix.py
python benchmarks/run_ablations.py
python benchmarks/run_decomposition.py

# 6. Work counting and query ef-semantics
python benchmarks/run_work_counting.py

# 7. Multi-thread build & filtered benchmark
python benchmarks/run_multithread_build.py
python benchmarks/run_filtered_bench.py

# 8. Self-retrieval measurements (engine and baselines)
python scripts/measure_fashion_mnist_self_retrieval.py
python benchmarks/measure_baseline_self_retrieval.py

# 9. Interpolations & stability evaluations
python benchmarks/equal_recall.py
python benchmarks/run_ranking_stability.py
python scripts/where_it_loses.py

# 10. Scorecard & documentation generation
python scripts/make_scorecard.py
python scripts/render_docs.py
python scripts/make_resume_bullets.py

# 11. Run complete verification and provenance checkers
python scripts/check_targets_unchanged.py
python scripts/check_claims.py
python scripts/check_provenance.py
python scripts/check_docs_links.py
python scripts/check_repro_commands.py
python scripts/verify_readme.py
```
