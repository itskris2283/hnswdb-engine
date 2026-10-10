import os
import sys
import tempfile
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from scripts.check_claims import check_claims, parse_allowlist


def test_banned_phrase_fails():
    with tempfile.TemporaryDirectory() as tmpdir:
        os.makedirs(os.path.join(tmpdir, "docs"))
        # Create empty allowlist
        with open(os.path.join(tmpdir, "docs", "claims_allowlist.txt"), "w") as f:
            f.write("# empty\n")
        
        # Test file with banned phrase
        bad_file = os.path.join(tmpdir, "README.md")
        with open(bad_file, "w", encoding="utf-8") as f:
            f.write("This engine uses lock-free concurrency.\n")

        errors = check_claims(root_dir=tmpdir)
        assert any("Contains prohibited phrase 'lock-free'" in e for e in errors)


def test_unscoped_allowlist_entry_fails():
    with tempfile.TemporaryDirectory() as tmpdir:
        os.makedirs(os.path.join(tmpdir, "docs"))
        al_path = os.path.join(tmpdir, "docs", "claims_allowlist.txt")
        with open(al_path, "w", encoding="utf-8") as f:
            f.write("unscoped_entry | 100% | reason\n")

        entries, errors = parse_allowlist(allowlist_path=al_path, raw_metric_vals=set(), target_thresholds=set())
        assert any("Unscoped allowlist entry" in e for e in errors)


def test_allowlisted_raw_metric_value_fails():
    with tempfile.TemporaryDirectory() as tmpdir:
        os.makedirs(os.path.join(tmpdir, "docs"))
        readme = os.path.join(tmpdir, "README.md")
        with open(readme, "w", encoding="utf-8") as f:
            f.write("Target is 0.9987 recall.\n")

        al_path = os.path.join(tmpdir, "docs", "claims_allowlist.txt")
        with open(al_path, "w", encoding="utf-8") as f:
            f.write("README.md:Target is | 0.9987 | attempting to allowlist measured metric\n")

        # Pass a set containing 0.9987 as a raw metric
        entries, errors = parse_allowlist(
            allowlist_path=al_path,
            raw_metric_vals={"0.9987"},
            target_thresholds=set()
        )
        assert any("matches raw JSON metric" in e for e in errors)


def test_missing_evidence_path_fails():
    with tempfile.TemporaryDirectory() as tmpdir:
        os.makedirs(os.path.join(tmpdir, "docs"))
        al_path = os.path.join(tmpdir, "docs", "claims_allowlist.txt")
        with open(al_path, "w") as f:
            f.write("# empty\n")

        inv_path = os.path.join(tmpdir, "docs", "INVESTIGATIONS.md")
        with open(inv_path, "w", encoding="utf-8") as f:
            f.write("We discovered the root cause of the bug.\n")
            f.write("evidence: non_existent_results_file.json\n")

        errors = check_claims(root_dir=tmpdir)
        assert any("Evidence path 'non_existent_results_file.json' does not exist" in e for e in errors)


def test_allowlist_cap_enforced():
    with tempfile.TemporaryDirectory() as tmpdir:
        os.makedirs(os.path.join(tmpdir, "docs"))
        al_path = os.path.join(tmpdir, "docs", "claims_allowlist.txt")
        with open(al_path, "w", encoding="utf-8") as f:
            # Write 26 entries
            for i in range(26):
                f.write(f"README.md:line_{i} | val_{i}_unit | reason {i}\n")

        entries, errors = parse_allowlist(allowlist_path=al_path, raw_metric_vals=set(), target_thresholds=set())
        assert any("exceeds 25-entry cap" in e for e in errors)
