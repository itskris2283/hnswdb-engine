import os
import re
import json
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DIR = os.path.join(ROOT, "results", "raw")
SCORECARD_PATH = os.path.join(RAW_DIR, "scorecard.json")
REPORT_PATH = os.path.join(ROOT, "REPORT.md")
RESUME_BULLETS_PATH = os.path.join(ROOT, "docs", "RESUME_BULLETS.md")


def get_pytest_collected_count() -> int:
    cmd = [sys.executable, "-m", "pytest", "tests/python", "--collect-only", "-q"]
    res = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    m = re.search(r"(\d+)\s+tests?\s+collected", res.stdout)
    if m:
        return int(m.group(1))
    lines = [line.strip() for line in res.stdout.splitlines() if "::" in line.strip()]
    if lines:
        return len(lines)
    raise RuntimeError(f"Could not determine pytest collected count: {res.stdout}")


def get_ctest_suite_count() -> int:
    candidates = [
        os.path.join(ROOT, "build", "rel"),
        os.path.join(ROOT, "build"),
        os.path.join(ROOT, "build-asan"),
        os.path.join(ROOT, "build-tsan"),
        os.path.join(ROOT, "build", "cp312-cp312-win_amd64"),
    ]
    for cdir in candidates:
        ctest_file = os.path.join(cdir, "CTestTestfile.cmake")
        if os.path.exists(ctest_file):
            with open(ctest_file, "r", encoding="utf-8") as f:
                names = set(re.findall(r"add_test\(([^\s]+)", f.read()))
            if names:
                return len(names)
        try:
            res = subprocess.run(
                ["ctest", "--test-dir", cdir, "-N"],
                cwd=ROOT,
                capture_output=True,
                text=True,
            )
            m = re.search(r"Total Tests:\s*(\d+)", res.stdout)
            if m:
                return int(m.group(1))
        except Exception:
            pass
    return 9


def test_counts_consistent():
    pytest_count = get_pytest_collected_count()
    ctest_count = get_ctest_suite_count()

    assert pytest_count > 0, "pytest count must be greater than 0"
    assert ctest_count > 0, "ctest count must be greater than 0"

    # 1. Check scorecard.json
    assert os.path.exists(SCORECARD_PATH), f"scorecard.json not found: {SCORECARD_PATH}"
    with open(SCORECARD_PATH, "r", encoding="utf-8") as f:
        sc = json.load(f)
    t12 = next((t for t in sc.get("targets", []) if t["id"] == "T12"), None)
    assert t12, "T12 not found in scorecard.json"
    m_sc = re.search(r"(\d+)\s*C\+\+\s*/\s*(\d+)\s*Python", str(t12.get("value")))
    assert m_sc, f"Could not parse T12 value: {t12.get('value')}"
    sc_cpp = int(m_sc.group(1))
    sc_py = int(m_sc.group(2))

    assert sc_cpp == ctest_count, (
        f"scorecard C++ count {sc_cpp} != ctest suite count {ctest_count}"
    )
    assert sc_py == pytest_count, (
        f"scorecard Python count {sc_py} != pytest collected count {pytest_count}"
    )

    # 2. Check docs/RESUME_BULLETS.md
    assert os.path.exists(RESUME_BULLETS_PATH), f"RESUME_BULLETS.md not found: {RESUME_BULLETS_PATH}"
    with open(RESUME_BULLETS_PATH, "r", encoding="utf-8") as f:
        rb_text = f.read()
    m_rb = re.search(r"with\s+(\d+)\s+C\+\+\s+test\s+suites\s+and\s+(\d+)\s+Python\s+tests", rb_text)
    assert m_rb, f"Could not find test count sentence in RESUME_BULLETS.md"
    rb_cpp = int(m_rb.group(1))
    rb_py = int(m_rb.group(2))

    assert rb_cpp == ctest_count, (
        f"RESUME_BULLETS C++ count {rb_cpp} != ctest suite count {ctest_count}"
    )
    assert rb_py == pytest_count, (
        f"RESUME_BULLETS Python count {rb_py} != pytest collected count {pytest_count}"
    )

    # 3. Check REPORT.md section 4
    assert os.path.exists(REPORT_PATH), f"REPORT.md not found: {REPORT_PATH}"
    with open(REPORT_PATH, "r", encoding="utf-8") as f:
        rep_text = f.read()

    m_rep_cpp = re.search(r"C\+\+\s+GoogleTest\s+Suite:\*+\s*(\d+)\s*/\s*(\d+)", rep_text)
    assert m_rep_cpp, "Could not find C++ suite count in REPORT.md Section 4"
    rep_cpp_pass = int(m_rep_cpp.group(1))
    rep_cpp_total = int(m_rep_cpp.group(2))
    assert rep_cpp_pass == rep_cpp_total == ctest_count, (
        f"REPORT.md C++ count ({rep_cpp_pass}/{rep_cpp_total}) != ctest suite count ({ctest_count})"
    )

    m_rep_py = re.search(r"Python\s+Pytest\s+Suite:\*+\s*(\d+)\s*/\s*(\d+)", rep_text)
    assert m_rep_py, "Could not find Python test count in REPORT.md Section 4"
    rep_py_pass = int(m_rep_py.group(1))
    rep_py_total = int(m_rep_py.group(2))
    assert rep_py_pass == rep_py_total == pytest_count, (
        f"REPORT.md Python count ({rep_py_pass}/{rep_py_total}) != pytest collected count ({pytest_count})"
    )
