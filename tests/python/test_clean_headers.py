import os
import glob
import re

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_no_generated_block_contains_unknown_or_question_marks():
    """Asserts that no generated block in README.md, REPORT.md, or docs/*.md contains 'unknown' or '? x ?'."""
    doc_paths = []
    for ext in ["*.md", os.path.join("docs", "*.md")]:
        doc_paths.extend(glob.glob(os.path.join(ROOT, ext)))

    assert len(doc_paths) > 0, "Must find documentation files to check"

    pattern = re.compile(r'<!--\s*BEGIN\s+GENERATED\s+table=(\S+)\s+run_id=("?[^">]+"|\S+)\s*-->(.*?)<!--\s*END\s+GENERATED\s*-->', re.DOTALL)

    violations = []
    for doc_path in doc_paths:
        rel_path = os.path.relpath(doc_path, ROOT)
        with open(doc_path, "r", encoding="utf-8") as f:
            content = f.read()

        for match in pattern.finditer(content):
            table_name = match.group(1)
            run_id = match.group(2).strip('"')
            block_body = match.group(3)

            if "unknown" in block_body.lower():
                violations.append(f"{rel_path} [{table_name} / {run_id}]: contains 'unknown'")
            if "? x ?" in block_body:
                violations.append(f"{rel_path} [{table_name} / {run_id}]: contains '? x ?'")

    assert not violations, "Found generated blocks with 'unknown' or '? x ?':\n" + "\n".join(violations)


def test_scorecard_header_format():
    """Asserts that the scorecard generated block header says 'Scorecard (multiple datasets)'."""
    report_path = os.path.join(ROOT, "REPORT.md")
    assert os.path.exists(report_path)
    with open(report_path, "r", encoding="utf-8") as f:
        content = f.read()

    pattern = re.compile(r'<!--\s*BEGIN\s+GENERATED\s+table=scorecard\s+run_id=("?[^">]+"|\S+)\s*-->(.*?)<!--\s*END\s+GENERATED\s*-->', re.DOTALL)
    match = pattern.search(content)
    assert match is not None, "REPORT.md must contain generated scorecard block"
    block_body = match.group(2)
    assert "**Scorecard (multiple datasets)**" in block_body, (
        "Scorecard block header must say '**Scorecard (multiple datasets)**'"
    )
