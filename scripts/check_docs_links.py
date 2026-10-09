#!/usr/bin/env python3
r"""
Documentation Link and Path Checker.
Verifies:

1. No absolute Windows paths (e.g. D:/..., C:\..., file:///) in documentation.
2. All relative markdown links and image paths exist on disk.
"""

import os
import re
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS_DIR = os.path.join(PROJECT_ROOT, "docs")
README_PATH = os.path.join(PROJECT_ROOT, "README.md")
REPORT_PATH = os.path.join(PROJECT_ROOT, "REPORT.md")


def check_docs_links():
    errors = []
    files_to_check = [README_PATH, REPORT_PATH]
    if os.path.exists(DOCS_DIR):
        for fname in os.listdir(DOCS_DIR):
            if fname.endswith(".md"):
                files_to_check.append(os.path.join(DOCS_DIR, fname))

    # Link regex: [text](path)
    link_pattern = re.compile(r'\[([^\]]+)\]\(([^)]+)\)')

    for fpath in files_to_check:
        if not os.path.exists(fpath):
            continue
        rel_fpath = os.path.relpath(fpath, PROJECT_ROOT)
        fdir = os.path.dirname(fpath)

        with open(fpath, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for line_num, line in enumerate(lines, 1):
            # Check for file:/// or Windows absolute drive paths (e.g., C:\ or D:/)
            if "file:///" in line or re.search(r'(?<!https)(?<!http)\b[a-zA-Z]:[\\/][a-zA-Z0-9_-]', line):
                errors.append(f"{rel_fpath}:{line_num}: Found prohibited absolute or file:/// path: {line.strip()}")


            # Check markdown links
            matches = link_pattern.findall(line)
            for text, target in matches:
                target = target.strip()
                # Skip web URLs, mailto, anchor links
                if target.startswith("http://") or target.startswith("https://") or target.startswith("#") or target.startswith("mailto:"):
                    continue

                # Strip anchor in path
                target_path = target.split("#")[0]
                if not target_path:
                    continue

                # Resolve relative target
                resolved = os.path.normpath(os.path.join(fdir, target_path))
                if not os.path.exists(resolved):
                    errors.append(f"{rel_fpath}:{line_num}: Broken link '{target}' (resolved to {os.path.relpath(resolved, PROJECT_ROOT)})")

    return errors


def main():
    print("Checking documentation links and paths...")
    errors = check_docs_links()
    if errors:
        print(f"FAILED: Found {len(errors)} broken links or forbidden paths:")
        for e in errors:
            print(f"  - {e}")
        return 1
    print("PASSED: All documentation links and relative paths resolve successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
