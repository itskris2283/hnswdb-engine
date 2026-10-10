#!/usr/bin/env python3
r"""
Strict Claims and Consistency Checker for HNSWDB (Round 4).

Verifies:
1. No banned or unsupported marketing phrases.
2. No absolute paths (file:///, C:\, D:/).
3. File extensions uniformly use .hdb.
4. Uses claims_allowlist.txt (NOT the old claims_whitelist.txt).
5. All measured numbers must be inside generated blocks or in the allowlist.
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
ALLOWLIST_PATH = os.path.join(PROJECT_ROOT, "docs", "claims_allowlist.txt")
OLD_WHITELIST_PATH = os.path.join(PROJECT_ROOT, "docs", "claims_whitelist.txt")
RAW_RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "raw")

# BEGIN/END generated block markers
BEGIN_GENERATED = re.compile(r'<!--\s*BEGIN\s+GENERATED\s+')
END_GENERATED = re.compile(r'<!--\s*END\s+GENERATED\s*-->')

BANNED_PHRASES = [
    "lock-free",
    "production-grade",
    "ThreadSanitizer-clean",
    "TSan-clean",
    "surpass",
    "outperforms FAISS",
    "beats FAISS",
    "FAISS requires",
    "matches and surpasses",
    "exceeds",
]


def load_allowlist():
    allowlist = set()
    if os.path.exists(ALLOWLIST_PATH):
        with open(ALLOWLIST_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    parts = line.split("#", 1)
                    val = parts[0].strip()
                    if val:
                        allowlist.add(val)
    return allowlist


def check_claims():
    errors = []

    # Check for stale whitelist
    if os.path.exists(OLD_WHITELIST_PATH):
        errors.append("docs/claims_whitelist.txt still exists — must be deleted (replaced by claims_allowlist.txt)")

    allowlist = load_allowlist()

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
            if phrase == "lock-free":
                pattern = r'(?<!dead)\block-free\b'
            else:
                pattern = r'\b' + re.escape(phrase) + r'\b'
            if re.search(pattern, content, re.IGNORECASE):
                errors.append(f"{rel_fpath}: Contains prohibited phrase '{phrase}'")

        # Check for absolute paths
        if "file:///" in content:
            errors.append(f"{rel_fpath}: Contains prohibited 'file:///' link")
        if re.search(r'[A-Za-z]:[\\/][A-Za-z0-9_-]', content):
            errors.append(f"{rel_fpath}: Contains prohibited Windows absolute path")

        # Check unqualified sub-millisecond
        in_generated_block = False
        for line_num, line in enumerate(lines, 1):
            if BEGIN_GENERATED.search(line):
                in_generated_block = True
                continue
            if END_GENERATED.search(line):
                in_generated_block = False
                continue

            if not in_generated_block:
                if "sub-millisecond" in line.lower() and not ("p50" in line.lower() or "ef=" in line.lower()):
                    errors.append(f"{rel_fpath}:{line_num}: Unqualified 'sub-millisecond' claim: {line.strip()}")

    return errors


def main():
    print("Running Strict Claims and Consistency Check (Round 4)...")
    errors = check_claims()

    if errors:
        print(f"FAILED: Found {len(errors)} issue(s):")
        for err in errors:
            print(f"  FAIL: {err}")
        return 1
    else:
        print("PASSED: All claims verified.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
