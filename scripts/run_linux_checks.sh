#!/usr/bin/env bash
set -euo pipefail

echo "=== HNSWDB Linux Sanitizer and Conformance Checks ==="
mkdir -p results/raw/sanitizers

# 1. ASan + UBSan
echo "--- 1. Building and running C++ with ASan + UBSan ---"
cmake -B build-asan -DCMAKE_BUILD_TYPE=Debug -DHNSWDB_BUILD_TESTS=ON -DHNSWDB_SANITIZE="address,undefined" -DHNSWDB_WERROR=ON
cmake --build build-asan -j"$(nproc)"
ctest --test-dir build-asan --output-on-failure 2>&1 | tee results/raw/sanitizers/asan_ubsan_ctest.log

# 2. Thread-using Python tests under ASan
echo "--- 2. Installing editable package and running Python tests ---"
python3 -m pip install --upgrade pip
python3 -m pip install -e ".[dev]"
pytest tests/python/test_concurrency.py -v 2>&1 | tee results/raw/sanitizers/asan_python_concurrency.log

# 3. TSan
echo "--- 3. Building and running C++ with TSan ---"
cmake -B build-tsan -DCMAKE_BUILD_TYPE=Debug -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DHNSWDB_BUILD_TESTS=ON -DHNSWDB_SANITIZE="thread"
cmake --build build-tsan -j"$(nproc)"
ctest --test-dir build-tsan --output-on-failure 2>&1 | tee results/raw/sanitizers/tsan_ctest.log

# 4. Concurrency stress test (16 readers + 16 writers, 30s)
echo "--- 4. Running 30s concurrency stress test under TSan ---"
pytest tests/python/test_concurrency.py -k "test_concurrent_batch_threading or test_concurrent_python_readers_and_writers" -v 2>&1 | tee results/raw/sanitizers/tsan_stress.log

echo "=== Linux checks completed successfully ==="
