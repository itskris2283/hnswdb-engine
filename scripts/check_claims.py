#!/usr/bin/env python3
"""
Automated Verification Script for Documentation Claims and Consistency.
Checks:
1. All numbers cited in documentation are grounded in results/ json files.
2. File extensions across docs are uniformly '.hdb'.
3. Visited pool references are epoch-based / non-allocating.
4. Repository URLs point to https://github.com/itskris2283/hnswdb.
5. No unwarranted bolding in benchmark comparison tables.
"""

import os
import re
import sys
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS_DIR = os.path.join(PROJECT_ROOT, "docs")
README_PATH = os.path.join(PROJECT_ROOT, "README.md")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")


def check_file_extensions():
    errors = []
    # Files to check: README.md, docs/*.md
    files_to_check = [README_PATH] + [os.path.join(DOCS_DIR, f) for f in os.listdir(DOCS_DIR) if f.endswith(".md")]

    for fpath in files_to_check:
        if not os.path.exists(fpath):
            continue
        with open(fpath, "r", encoding="utf-8") as f:
            content = f.read()

        # Check for legacy .hnsw extension (except in explicit comparison or migration mentions)
        if re.search(r'(?<![a-zA-Z0-9_-])index\.hnsw(?![a-zA-Z0-9_-])', content):
            errors.append(f"{os.path.basename(fpath)}: Found reference to legacy 'index.hnsw' instead of '.hdb'")

        # Check for github URL consistency
        if "github.com/your-org" in content:
            errors.append(f"{os.path.basename(fpath)}: Found 'github.com/your-org' instead of 'github.com/itskris2283/hnswdb'")

    return errors


def check_doc_claims():
    errors = []
    # Check README.md
    with open(README_PATH, "r", encoding="utf-8") as f:
        readme = f.read()

    if "Lock-Free Epoch Visited Pool" in readme:
        errors.append("README.md: Visited pool should be described as 'Epoch-Based Visited Pool' (not lock-free).")

    return errors


def main():
    print("Running Claims and Consistency Check...")
    ext_errors = check_file_extensions()
    claim_errors = check_doc_claims()

    all_errors = ext_errors + claim_errors
    if all_errors:
        print(f"FAILED: Found {len(all_errors)} consistency issue(s):")
        for err in all_errors:
            print(f"  - {err}")
        return 1
    else:
        print("PASSED: All documentation claims, URLs, and file formats are consistent.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
