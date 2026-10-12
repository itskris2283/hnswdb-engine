#!/usr/bin/env python3
"""
scripts/check_repro_commands.py — Validate reproduction and testing commands.

Extracts every python script command from the reproduction blocks in REPORT.md and README.md,
verifies that each referenced script path exists, and verifies that it passes py_compile.
"""

import os
import py_compile
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def extract_python_scripts_from_block(text: str) -> list[str]:
    """Finds referenced python scripts in shell code blocks."""
    scripts = []
    # Match lines like: python path/to/script.py [args] or python -m ...
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # match python [flags] <script.py>
        m = re.search(r'\bpython(?:\.exe)?\s+(?:-[a-zA-Z0-9_]+\s+)*([a-zA-Z0-9_./\\]+\.py)\b', line)
        if m:
            script_path = m.group(1).replace("\\", "/")
            scripts.append(script_path)
    return scripts


def check_file_repro_blocks(filepath: str, section_header: str = None) -> list[tuple[str, str]]:
    """Extract code blocks in the specified section (or whole file) and check all python scripts."""
    if not os.path.exists(filepath):
        return [("FILE_NOT_FOUND", filepath)]

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    section_text = content
    if section_header:
        idx = content.find(section_header)
        if idx == -1:
            return [("SECTION_NOT_FOUND", f"{filepath}: {section_header}")]

        section_text = content[idx:]
        next_sec = re.search(r'\n##\s+', section_text[len(section_header):])
        if next_sec:
            section_text = section_text[:len(section_header) + next_sec.start()]

    # Extract ```bash or ``` blocks
    code_blocks = re.findall(r'```(?:bash|sh)?\s*\n(.*?)\n```', section_text, re.DOTALL)
    if not code_blocks:
        return [("NO_CODE_BLOCK", filepath)]

    scripts = []
    for cb in code_blocks:
        scripts.extend(extract_python_scripts_from_block(cb))

    return [("SCRIPT", s) for s in scripts]


def main():
    print("Checking reproduction commands in docs/REPRODUCE.md...")
    errors = []
    checked = set()

    reproduce_path = os.path.join(ROOT, "docs", "REPRODUCE.md")
    rel_f = os.path.relpath(reproduce_path, ROOT)
    results = check_file_repro_blocks(reproduce_path, None)
    for kind, val in results:
        if kind == "SCRIPT":
            script_rel = val
            script_full = os.path.join(ROOT, script_rel)
            if not os.path.exists(script_full):
                errors.append(f"{rel_f}: Script does not exist: '{script_rel}'")
            else:
                if script_full not in checked:
                    checked.add(script_full)
                    try:
                        py_compile.compile(script_full, doraise=True)
                    except py_compile.PyCompileError as e:
                        errors.append(f"{rel_f}: Compilation error in '{script_rel}': {e}")
        else:
            errors.append(f"{rel_f}: {kind} ({val})")

    if errors:
        print("\nFAILED: Found issues with reproduction commands:")
        for err in errors:
            print(f"  FAIL: {err}")
        sys.exit(1)

    print(f"PASSED: All {len(checked)} referenced reproduction script(s) exist and compile successfully.")
    sys.exit(0)


if __name__ == "__main__":
    main()
