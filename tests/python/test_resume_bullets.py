import os
import re
import json
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESUME_BULLETS_PATH = os.path.join(ROOT, "docs", "RESUME_BULLETS.md")


def test_resume_bullets_file_exists():
    assert os.path.exists(RESUME_BULLETS_PATH), "docs/RESUME_BULLETS.md must exist"


def test_resume_bullets_provenance():
    with open(RESUME_BULLETS_PATH, "r", encoding="utf-8") as f:
        content = f.read()

    # Verify no forbidden speed comparisons
    lower = content.lower()
    assert "faster than" not in lower, "No bullet may claim HNSWDB is faster than another engine"
    assert "speed ratio" not in lower, "No bullet may claim a speed ratio"
    assert "beats faiss" not in lower
    assert "beats hnswlib" not in lower

    # Parse provenance table rows
    table_lines = [line.strip() for line in content.splitlines() if line.strip().startswith("| `")]
    assert len(table_lines) >= 9, f"Provenance table must have at least 9 metric rows, got {len(table_lines)}"

    prov_entries = {}
    for line in table_lines:
        parts = [p.strip() for p in line.strip("|").split("|")]
        if len(parts) >= 5:
            placeholder = parts[0].strip("`")
            rendered_val = parts[1]
            raw_file = parts[2].strip("`")
            key_path = parts[3]
            run_id = parts[4].strip("`")
            prov_entries[placeholder] = {
                "val": rendered_val,
                "file": raw_file,
                "key_path": key_path,
                "run_id": run_id,
            }

    # Verify every provenance row against the real raw JSON
    for placeholder, entry in prov_entries.items():
        abs_path = os.path.join(ROOT, entry["file"])
        assert os.path.exists(abs_path), f"Raw file {abs_path} must exist"
        with open(abs_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        if placeholder == "n_cpp":
            t12 = next(t for t in raw_data["targets"] if t["id"] == "T12")
            assert str(entry["val"]) in t12["value"]
        elif placeholder == "n_py":
            t12 = next(t for t in raw_data["targets"] if t["id"] == "T12")
            assert str(entry["val"]) in t12["value"]
        elif placeholder == "fm_recall50":
            h = raw_data["engines"]["hnswdb"]["seed_42"]
            r50 = next(r for r in h["ef_results"] if r["ef_search"] == 50)
            assert f"{r50['recall_at_k']:.4f}" == entry["val"]
        elif placeholder == "fm_p50":
            h = raw_data["engines"]["hnswdb"]["seed_42"]
            r50 = next(r for r in h["ef_results"] if r["ef_search"] == 50)
            assert f"{r50['latency']['p50']:.3f}" == entry["val"]
        elif placeholder == "legacy_recall_mean":
            rec = raw_data["engines"]["hnswdb_legacy"]["recall_1t_mean"]
            assert f"{rec:.4f}" == entry["val"]
        elif placeholder == "cured_recall_mean":
            rec = raw_data["engines"]["hnswdb_cured"]["recall_1t_mean"]
            assert f"{rec:.4f}" == entry["val"]
        elif placeholder == "legacy_recall_min":
            rec = raw_data["engines"]["hnswdb_legacy"]["recall_1t_min"]
            assert f"{rec:.4f}" == entry["val"]
        elif placeholder == "cured_recall_min":
            rec = raw_data["engines"]["hnswdb_cured"]["recall_1t_min"]
            assert f"{rec:.4f}" == entry["val"]
        elif placeholder == "legacy_unreach_mean":
            unreach = raw_data["engines"]["hnswdb_legacy"]["unreachable_mean"]
            assert f"{unreach:.2f}" == entry["val"]
        elif placeholder == "cured_unreach_mean":
            unreach = raw_data["engines"]["hnswdb_cured"]["unreachable_mean"]
            assert f"{unreach:.2f}" == entry["val"]
        elif placeholder == "n_targets":
            assert len(raw_data["targets"]) == int(entry["val"])

    # Verify that the rendered bullet lines contain all rendered values
    bullet_lines = [l for l in content.splitlines() if l.strip().startswith("* ")]
    assert len(bullet_lines) == 3, f"Must have exactly 3 bullet points, got {len(bullet_lines)}"
    combined_bullets = " ".join(bullet_lines)

    for placeholder, entry in prov_entries.items():
        assert entry["val"] in combined_bullets, (
            f"Rendered value {entry['val']} for {placeholder} must appear in the bullet points"
        )
