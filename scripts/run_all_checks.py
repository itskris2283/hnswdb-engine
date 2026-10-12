"""
Run full verification suite (CTest, PyTest, check_claims, check_docs_links)
and output comprehensive log to results/raw/run_all_checks.txt.
"""

import os
import sys
import subprocess
import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_FILE = os.path.join(PROJECT_ROOT, "results", "raw", "run_all_checks.txt")


def run_command(cmd, cwd=PROJECT_ROOT):
    print(f"=== Running: {' '.join(cmd)} ===")
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    out = f"=== COMMAND: {' '.join(cmd)} ===\nExit Code: {r.returncode}\n--- STDOUT ---\n{r.stdout}\n--- STDERR ---\n{r.stderr}\n\n"
    return r.returncode, out


def main():
    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    all_logs = []
    all_logs.append(f"HNSWDB Full Verification Run - {datetime.datetime.now().isoformat()}\n")

    overall_pass = True

    # 1. CTest
    test_dir = "build/cp312-cp312-win_amd64" if os.path.exists(os.path.join(PROJECT_ROOT, "build", "cp312-cp312-win_amd64", "CTestTestfile.cmake")) else "build/rel"
    code, out = run_command(["ctest", "--test-dir", test_dir, "-C", "Release", "--output-on-failure"])
    all_logs.append(out)
    if code != 0: overall_pass = False

    # 2. PyTest
    code, out = run_command([sys.executable, "-m", "pytest", "tests/python", "-v"])
    all_logs.append(out)
    if code != 0: overall_pass = False

    # 3. Check Targets Unchanged
    code, out = run_command([sys.executable, "scripts/check_targets_unchanged.py"])
    all_logs.append(out)
    if code != 0: overall_pass = False

    # 4. Check Claims
    claims_out = os.path.join(PROJECT_ROOT, "results", "raw", "check_claims.txt")
    code, out = run_command([sys.executable, "scripts/check_claims.py", "--output", claims_out])
    all_logs.append(out)
    if code != 0: overall_pass = False

    # 5. Check Provenance
    prov_out = os.path.join(PROJECT_ROOT, "results", "raw", "provenance_check.txt")
    code, out = run_command([sys.executable, "scripts/check_provenance.py", "--output", prov_out])
    all_logs.append(out)
    if code != 0: overall_pass = False

    # 6. Check Docs Links
    code, out = run_command([sys.executable, "scripts/check_docs_links.py"])
    all_logs.append(out)
    if code != 0: overall_pass = False

    # 7. Check Reproduction Commands
    code, out = run_command([sys.executable, "scripts/check_repro_commands.py"])
    all_logs.append(out)
    if code != 0: overall_pass = False

    # 8. Verify README
    code, out = run_command([sys.executable, "scripts/verify_readme.py"])
    all_logs.append(out)
    if code != 0: overall_pass = False

    # 9. Test Counts Consistency
    code, out = run_command([sys.executable, "-m", "pytest", "tests/python/test_counts_consistent.py", "-v"])
    all_logs.append(out)
    if code != 0: overall_pass = False

    summary = f"OVERALL STATUS: {'ALL CHECKS PASSED' if overall_pass else 'FAILED'}\n"
    all_logs.append(summary)

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        f.writelines(all_logs)

    print(f"Full check log written to: {OUT_FILE}")
    print(summary)
    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
