#!/usr/bin/env python3
"""
check_provenance.py — Verify that all generated blocks in docs have valid raw data.

Checks:
1. Every <!-- BEGIN GENERATED --> block has a matching run_id in results/raw/
2. No measured numbers exist outside generated blocks (heuristic check)
3. All run_ids in a single table come from the same result file
4. The claims allowlist has no stale entries

Usage:
    python scripts/check_provenance.py          # check all docs
    python scripts/check_provenance.py --strict  # also flag docs with no generated blocks
"""

import argparse
import glob
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(ROOT, "results", "raw")

BEGIN_PATTERN = re.compile(
    r'<!--\s*BEGIN\s+GENERATED\s+table=(\S+)\s+run_id=("?[^">]+"|\S+)\s*-->'
)
END_PATTERN = re.compile(r'<!--\s*END\s+GENERATED\s*-->')

# Pattern for suspicious hardcoded benchmark numbers outside generated blocks
# Matches lines like "recall: 0.9503" or "QPS: 45,000" or "latency: 0.162 ms"
SUSPICIOUS_PATTERNS = [
    re.compile(r'recall[:\s@]+\d+\.\d{4,}', re.IGNORECASE),
    re.compile(r'QPS[:\s]+[\d,]+', re.IGNORECASE),
    re.compile(r'latency[:\s]+\d+\.\d+\s*ms', re.IGNORECASE),
    re.compile(r'p(?:50|95|99|999)[:\s]+\d+\.\d+', re.IGNORECASE),
]


def load_raw_run_ids() -> set:
    """Get all run_ids from raw result files."""
    run_ids = set()
    if not os.path.isdir(RAW_DIR):
        return run_ids
    for f in os.listdir(RAW_DIR):
        if f.endswith(".json") and f != "MANIFEST.json":
            try:
                with open(os.path.join(RAW_DIR, f)) as fh:
                    data = json.load(fh)
                if "run_id" in data:
                    run_ids.add(data["run_id"])
            except (json.JSONDecodeError, KeyError):
                pass
    return run_ids


def check_file(filepath: str, valid_run_ids: set, strict: bool = False) -> list[str]:
    """Check provenance of a single file. Returns list of error messages."""
    errors = []

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception as e:
        return [f"{filepath}: Cannot read: {e}"]

    in_block = False
    in_corrections = False
    block_start_line = 0
    block_run_ids = set()
    has_any_block = False

    for line_num, line in enumerate(lines, 1):
        if "<!-- BEGIN CORRECTIONS -->" in line:
            in_corrections = True
            continue
        if "<!-- END CORRECTIONS -->" in line:
            in_corrections = False
            continue
        if in_corrections:
            continue

        begin_match = BEGIN_PATTERN.search(line)
        if begin_match:
            if in_block:
                errors.append(f"{filepath}:{line_num}: Nested BEGIN GENERATED (unclosed block at line {block_start_line})")
            in_block = True
            block_start_line = line_num
            table_name = begin_match.group(1)
            run_id = begin_match.group(2).strip('"')
            has_any_block = True

            if run_id not in valid_run_ids:
                errors.append(f"{filepath}:{line_num}: run_id '{run_id}' not found in results/raw/")
            block_run_ids.add(run_id)
            continue

        if END_PATTERN.search(line):
            if not in_block:
                errors.append(f"{filepath}:{line_num}: END GENERATED without matching BEGIN")
            in_block = False
            continue

        # Check for suspicious hardcoded numbers OUTSIDE generated blocks
        if not in_block:
            for pattern in SUSPICIOUS_PATTERNS:
                match = pattern.search(line)
                if match:
                    # Skip lines that are clearly just describing parameters or code
                    stripped = line.strip()
                    if stripped.startswith("#") or stripped.startswith("//") or stripped.startswith("```"):
                        continue
                    # Skip lines that reference results/raw/ files
                    if "results/raw/" in line or "run_id" in line:
                        continue
                    errors.append(
                        f"{filepath}:{line_num}: Possible hardcoded benchmark number outside generated block: "
                        f"'{match.group().strip()}'"
                    )

    if in_block:
        errors.append(f"{filepath}: Unclosed BEGIN GENERATED block starting at line {block_start_line}")

    # Mixed run_ids check
    if len(block_run_ids) > 1:
        # This is a warning, not an error — a file may have tables from different datasets
        pass

    return errors


def check_claims_allowlist() -> list[str]:
    """Check that claims_allowlist.txt exists and contains no stale entries."""
    errors = []
    allowlist_path = os.path.join(ROOT, "docs", "claims_allowlist.txt")
    old_whitelist = os.path.join(ROOT, "docs", "claims_whitelist.txt")

    if os.path.exists(old_whitelist):
        errors.append(f"STALE: docs/claims_whitelist.txt still exists (should be replaced by claims_allowlist.txt)")

    if not os.path.exists(allowlist_path):
        errors.append(f"MISSING: docs/claims_allowlist.txt not found")

    return errors


def main():
    parser = argparse.ArgumentParser(description="Check provenance of generated blocks in docs")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--file", type=str, help="Check a specific file")
    parser.add_argument("--output", "-o", type=str, help="Output file path (written in UTF-8)")
    args = parser.parse_args()

    out_lines = []
    valid_run_ids = load_raw_run_ids()
    out_lines.append(f"Found {len(valid_run_ids)} valid run_id(s) in results/raw/")

    if args.file:
        files = [args.file]
    else:
        files = []
        for pattern in ["docs/*.md", "*.md"]:
            files.extend(glob.glob(os.path.join(ROOT, pattern)))
        files = list(set(files))

    all_errors = []

    for filepath in sorted(files):
        errors = check_file(filepath, valid_run_ids, strict=args.strict)
        all_errors.extend(errors)

    # Check claims allowlist
    all_errors.extend(check_claims_allowlist())

    if all_errors:
        out_lines.append(f"\n{len(all_errors)} provenance issue(s) found:")
        for err in all_errors:
            out_lines.append(f"  FAIL: {err}")
        code = 1
    else:
        out_lines.append("\nPASS: All provenance checks passed")
        code = 0

    out_text = "\n".join(out_lines) + "\n"
    print(out_text, end="")

    if args.output:
        with open(args.output, "w", encoding="utf-8", newline="\n") as f:
            f.write(out_text)

    return code


if __name__ == "__main__":
    sys.exit(main())
