import os
import re
import json
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DIR = os.path.join(ROOT, "results", "raw")
README_PATH = os.path.join(ROOT, "README.md")


def extract_where_it_loses_table_rows():
    with open(README_PATH, "r", encoding="utf-8") as f:
        content = f.read()

    match = re.search(
        r"<!-- BEGIN GENERATED table=where_it_loses run_id=([^\s>]+) -->\s*(.*?)\s*<!-- END GENERATED -->",
        content,
        re.DOTALL,
    )
    assert match, "where_it_loses block not found in README.md"

    table_body = match.group(2).strip()
    lines = [line.strip() for line in table_body.splitlines() if line.strip()]

    # Header and separator
    assert len(lines) >= 3, "Table should have header, separator, and data rows"
    header = lines[0]
    separator = lines[1]
    assert "Workload" in header and "Metric" in header and "Source" in header

    rows = []
    for line in lines[2:]:
        if not line.startswith("|"):
            continue
        cols = [c.strip() for c in line.split("|")[1:-1]]
        if len(cols) == 6:
            rows.append({
                "workload": cols[0],
                "metric": cols[1],
                "hnswdb": cols[2],
                "best_baseline": cols[3],
                "gap": cols[4],
                "source": cols[5].strip("` "),
            })
    return rows


def test_where_it_loses_numbers_appear_in_source():
    rows = extract_where_it_loses_table_rows()
    assert len(rows) > 0, "Expected at least one losing comparison row"

    for row in rows:
        source_str = row["source"]
        source_parts = [p.strip() for p in source_str.split(";")]
        all_raw_text = ""
        for sp in source_parts:
            raw_filename = sp.split(":")[0].strip()
            raw_filepath = os.path.join(RAW_DIR, raw_filename)
            assert os.path.exists(raw_filepath), f"Source file does not exist: {raw_filepath}"
            with open(raw_filepath, "r", encoding="utf-8") as rf:
                all_raw_text += rf.read()

        # Extract numeric tokens from hnswdb and best_baseline
        for col_name in ["hnswdb", "best_baseline"]:
            val_str = row[col_name]
            num_matches = re.findall(r"\d+(?:\.\d+)?", val_str)
            assert num_matches, f"No number found in {col_name}: {val_str}"
            for num in num_matches:
                # The exact number must appear in the raw JSON file
                assert num in all_raw_text, (
                    f"Number {num} from {col_name}='{val_str}' not found in raw files for {source_str}"
                )


def test_where_it_loses_matches_independent_recomputation():
    from scripts.where_it_loses import analyze_losing_comparisons

    expected_comparisons, _ = analyze_losing_comparisons()
    table_rows = extract_where_it_loses_table_rows()

    assert len(table_rows) == len(expected_comparisons), (
        f"Row count mismatch: {len(table_rows)} in README vs {len(expected_comparisons)} recomputed"
    )

    for exp, row in zip(expected_comparisons, table_rows):
        assert row["workload"] == exp["workload"]
        assert row["metric"] == exp["metric"]
        assert row["hnswdb"] == exp["hnswdb"]
        assert row["best_baseline"] == exp["best_baseline"]
        assert row["gap"] == f"**{exp['gap']}**"
        assert row["source"] == exp["source"]


def test_where_it_loses_includes_self_retrieval():
    rows = extract_where_it_loses_table_rows()
    self_retrieval_rows = [
        r for r in rows
        if "self-retrieval" in r["metric"].lower() or "unreachable" in r["metric"].lower()
    ]
    assert len(self_retrieval_rows) >= 2, (
        f"Expected at least 2 self-retrieval failure rows (Dataset B and Fashion-MNIST), found {len(self_retrieval_rows)}"
    )
    workloads = [r["workload"] for r in self_retrieval_rows]
    assert any("Dataset B" in w for w in workloads), "Dataset B self-retrieval row missing from where_it_loses"
    assert any("Fashion-MNIST" in w for w in workloads), "Fashion-MNIST self-retrieval row missing from where_it_loses"
