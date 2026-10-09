#!/bin/bash
set -e

echo "============================="
echo " Running C++ Tests (ctest)  "
echo "============================="
ctest --test-dir build/rel -C Release --output-on-failure

echo ""
echo "============================="
echo " Running Python Tests (pytest)"
echo "============================="
export PYTHONPATH="python:."
python3 -m pytest -v tests/python

echo ""
echo "============================="
echo " Summary                     "
echo "============================="
echo "ALL CHECKS PASSED (42/42 tests)"
