#!/usr/bin/env python3
"""
tests/python/test_text_encoding.py — Assert that all documentation and result files are valid UTF-8.

Asserts that every .md, .txt, and .json file under docs/, results/raw/ and the repo root
is valid UTF-8 without BOM and contains no null bytes.
"""

import glob
import os
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_files_to_check() -> list[str]:
    files = []
    # Root .md, .txt, .json
    for ext in ["*.md", "*.txt", "*.json"]:
        files.extend(glob.glob(os.path.join(ROOT, ext)))

    # docs/ recursively
    for dirpath, _, filenames in os.walk(os.path.join(ROOT, "docs")):
        for fn in filenames:
            if fn.endswith((".md", ".txt", ".json")):
                files.append(os.path.join(dirpath, fn))

    # results/raw/ recursively
    for dirpath, _, filenames in os.walk(os.path.join(ROOT, "results", "raw")):
        for fn in filenames:
            if fn.endswith((".md", ".txt", ".json")):
                files.append(os.path.join(dirpath, fn))

    return sorted(set(files))


def test_all_text_files_are_valid_utf8_without_bom():
    files = get_files_to_check()
    assert len(files) > 0, "Must find files to test"

    failures = []
    for fpath in files:
        rel = os.path.relpath(fpath, ROOT).replace("\\", "/")
        try:
            with open(fpath, "rb") as fh:
                raw = fh.read()
        except Exception as e:
            failures.append(f"{rel}: Cannot open file: {e}")
            continue

        if b"\x00" in raw:
            failures.append(f"{rel}: Contains null bytes (not valid text/UTF-8)")

        if raw.startswith(b"\xef\xbb\xbf"):
            failures.append(f"{rel}: Contains UTF-8 BOM (must be UTF-8 without BOM)")

        if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
            failures.append(f"{rel}: Contains UTF-16 BOM (must be UTF-8 without BOM)")

        try:
            raw.decode("utf-8")
        except UnicodeDecodeError as e:
            failures.append(f"{rel}: UnicodeDecodeError when decoding as UTF-8: {e}")

    assert not failures, "Encoding failures found:\n" + "\n".join(failures)
