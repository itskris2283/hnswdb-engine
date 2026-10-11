#!/usr/bin/env python3
"""
Test suite for scripts/check_repro_commands.py.
"""

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_reproduction_commands_pass():
    script_path = os.path.join(ROOT, "scripts", "check_repro_commands.py")
    res = subprocess.run([sys.executable, script_path], cwd=ROOT, capture_output=True, text=True)
    assert res.returncode == 0, f"check_repro_commands.py failed:\n{res.stdout}\n{res.stderr}"
    assert "PASSED: All" in res.stdout
