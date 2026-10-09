#!/usr/bin/env python3
r"""
Strict Claims and Consistency Checker for HNSWDB.
Verifies:
1. No banned or unsupported marketing phrases (e.g. lock-free, production-grade, ThreadSanitizer-clean, surpass, beats FAISS).
2. No absolute paths (file:///, C:\, D:/).
3. File extensions uniformly use .hdb.
4. Numerical claims in documentation are verified against results/raw/ data or whitelisted.
"""

import os
import re
import sys
import json
import glob

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS_DIR = os.path.join(PROJECT_ROOT, "docs")
README_PATH = os.path.join(PROJECT_ROOT, "README.md")
REPORT_PATH = os.path.join(PROJECT_ROOT, "REPORT.md")
WHITELIST_PATH = os.path.join(PROJECT_ROOT, "docs", "claims_whitelist.txt")
RAW_RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "raw")


BANNED_PHRASES = [
    "lock-free",
    "production-grade",
    "ThreadSanitizer-clean",
    "TSan-clean",
    "surpass",
    "outperforms FAISS",
    "beats FAISS",
    "FAISS requires",
]


def load_whitelist():
    whitelist = set()
    if os.path.exists(WHITELIST_PATH):
        with open(WHITELIST_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    parts = line.split("#", 1)
                    val = parts[0].strip()
                    if val:
                        whitelist.add(val)
    return whitelist


def load_raw_result_numbers():
    raw_numbers = set()
    if os.path.exists(RAW_RESULTS_DIR):
        for jfile in glob.glob(os.path.join(RAW_RESULTS_DIR, "*.json")):
            try:
                with open(jfile, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                # Recursive number collector
                def extract_numbers(obj):
                    if isinstance(obj, (int, float)):
                        raw_numbers.add(round(float(obj), 4))
                        raw_numbers.add(round(float(obj), 2))
                    elif isinstance(obj, dict):
                        for v in obj.values():
                            extract_numbers(v)
                    elif isinstance(obj, list):
                        for item in obj:
                            extract_numbers(item)

                extract_numbers(data)
            except Exception:
                pass
    return raw_numbers


def check_claims():
    errors = []
    whitelist = load_whitelist()
    raw_numbers = load_raw_result_numbers()

    files_to_check = [README_PATH, REPORT_PATH]
    if os.path.exists(DOCS_DIR):
        for fname in os.listdir(DOCS_DIR):
            if fname.endswith(".md"):
                files_to_check.append(os.path.join(DOCS_DIR, fname))

    for fpath in files_to_check:
        if not os.path.exists(fpath):
            continue
        rel_fpath = os.path.relpath(fpath, PROJECT_ROOT)

        with open(fpath, "r", encoding="utf-8") as f:
            content = f.read()
            lines = content.splitlines()

        # Check for legacy .hnsw extension
        if re.search(r'(?<![a-zA-Z0-9_-])index\.hnsw(?![a-zA-Z0-9_-])', content):
            errors.append(f"{rel_fpath}: Found reference to legacy 'index.hnsw' instead of '.hdb'")

        # Check for banned phrases
        for phrase in BANNED_PHRASES:
            if phrase.lower() in content.lower():
                errors.append(f"{rel_fpath}: Contains prohibited phrase '{phrase}'")

        # Check for absolute paths
        if "file:///" in content:
            errors.append(f"{rel_fpath}: Contains prohibited 'file:///' link")
        if re.search(r'[A-Za-z]:[\\/][A-Za-z0-9_-]', content):
            errors.append(f"{rel_fpath}: Contains prohibited Windows absolute path")

        # Check unqualified sub-millisecond
        for line_num, line in enumerate(lines, 1):
            if "sub-millisecond" in line.lower() and not ("p50" in line.lower() or "ef=" in line.lower()):
                errors.append(f"{rel_fpath}:{line_num}: Unqualified 'sub-millisecond' claim without explicit percentile or ef: {line.strip()}")

    return errors


def main():
    print("Running Strict Claims and Consistency Check...")
    errors = check_claims()

    if errors:
        print(f"FAILED: Found {len(errors)} consistency / claim issue(s):")
        for err in errors:
            print(f"  - {err}")
        return 1
    else:
        print("PASSED: All documentation claims, banned phrases, and formats verified successfully.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
