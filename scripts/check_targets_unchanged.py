#!/usr/bin/env python3
"""
check_targets_unchanged.py — Verify that benchmarks/targets.json matches round5-targets-frozen tag.

Ensures targets are completely frozen and cannot be modified to fit measurements.
"""

import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGETS_PATH = os.path.join(ROOT, "benchmarks", "targets.json")
TAG_NAME = "round5-targets-frozen"


def check_targets():
    if not os.path.exists(TARGETS_PATH):
        print(f"FAIL: {TARGETS_PATH} does not exist!")
        return 1

    with open(TARGETS_PATH, "r", encoding="utf-8") as f:
        try:
            working_data = json.load(f)
        except Exception as e:
            print(f"FAIL: {TARGETS_PATH} is not valid JSON: {e}")
            return 1

    # Check against git tag
    try:
        tag_check = subprocess.run(
            ["git", "rev-parse", "--verify", f"refs/tags/{TAG_NAME}"],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        if tag_check.returncode != 0:
            print(f"FAIL: Git tag '{TAG_NAME}' does not exist yet. Create and tag targets first.")
            return 1

        tagged_show = subprocess.run(
            ["git", "show", f"{TAG_NAME}:benchmarks/targets.json"],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        if tagged_show.returncode != 0:
            print(f"FAIL: Could not retrieve benchmarks/targets.json from tag '{TAG_NAME}': {tagged_show.stderr}")
            return 1

        tagged_data = json.loads(tagged_show.stdout)

        if working_data != tagged_data:
            print(f"FAIL: benchmarks/targets.json differs from frozen tag '{TAG_NAME}'!")
            print("Targets are frozen and must never be altered.")
            return 1

    except Exception as e:
        print(f"FAIL: Error checking git tag: {e}")
        return 1

    print(f"PASSED: benchmarks/targets.json strictly matches frozen tag '{TAG_NAME}'.")
    return 0


if __name__ == "__main__":
    sys.exit(check_targets())
