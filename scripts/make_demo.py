#!/usr/bin/env python3
"""
Full End-to-End System Demo Runner.
Executes quickstart, semantic search, and drop-in collection demos to verify all user-facing examples.
"""

import subprocess
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_script(rel_path):
    full_path = os.path.join(PROJECT_ROOT, rel_path)
    print(f"\n=======================================================")
    print(f"Running Demo: {rel_path}")
    print(f"=======================================================")
    res = subprocess.run([sys.executable, full_path], cwd=PROJECT_ROOT)
    if res.returncode != 0:
        print(f"ERROR: Demo {rel_path} failed with exit code {res.returncode}")
        return False
    return True


def main():
    demos = [
        "examples/quickstart.py",
        "examples/demo_semantic_search.py",
        "examples/chroma_swap.py",
    ]
    all_passed = True
    for demo in demos:
        if not run_script(demo):
            all_passed = False

    if all_passed:
        print("\nAll demos executed and passed successfully!")
        return 0
    else:
        print("\nSome demos failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
