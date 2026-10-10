#!/usr/bin/env python3
"""
render_docs.py — Render generated blocks in markdown docs from raw JSON data.

Scans all .md files in docs/ and the repo root for blocks delimited by:
    <!-- BEGIN GENERATED table=<name> run_id=<run_id> -->
    ...content...
    <!-- END GENERATED -->

Replaces the content between the markers with data from results/raw/<file>.json
that matches the run_id.

Usage:
    python scripts/render_docs.py                    # render all docs
    python scripts/render_docs.py --check            # dry-run: report what would change
    python scripts/render_docs.py --file docs/BENCHMARKS.md
"""

import argparse
import glob
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(ROOT, "results", "raw")

# Pattern for generated block markers
BEGIN_PATTERN = re.compile(
    r'<!--\s*BEGIN\s+GENERATED\s+table=(\S+)\s+run_id=(\S+)\s*-->'
)
END_PATTERN = re.compile(r'<!--\s*END\s+GENERATED\s*-->')


def load_raw_results() -> dict:
    """Load all raw JSON results indexed by run_id."""
    results = {}
    if not os.path.isdir(RAW_DIR):
        return results
    for f in os.listdir(RAW_DIR):
        if f.endswith(".json") and f != "MANIFEST.json":
            path = os.path.join(RAW_DIR, f)
            try:
                with open(path) as fh:
                    data = json.load(fh)
                if "run_id" in data:
                    results[data["run_id"]] = data
            except (json.JSONDecodeError, KeyError):
                pass
    return results


def format_recall_table(data: dict, table_name: str) -> str:
    """Format a recall/latency table from raw benchmark data."""
    lines = []

    engines = data.get("engines", {})
    params = data.get("params", {})
    dataset_info = data.get("dataset_info", {})

    # Header
    lines.append(f"**Dataset**: {data.get('dataset', 'unknown')} "
                 f"({dataset_info.get('n_train', '?')} x {dataset_info.get('dim', '?')}d) | "
                 f"**M**={params.get('M', '?')} | **efC**={params.get('ef_construction', '?')} | "
                 f"**k**={params.get('k', '?')}")
    lines.append("")

    if table_name.startswith("recall"):
        # Recall table
        lines.append("| ef_search | " + " | ".join(
            f"{eng} recall@{params.get('k', 10)}" for eng in engines
        ) + " |")
        lines.append("| ---: | " + " | ".join("---:" for _ in engines) + " |")

        # Get seed (use first available)
        for ef in params.get("ef_values", []):
            row = f"| {ef} |"
            for eng_name in engines:
                eng_data = engines[eng_name]
                # Find first seed's data
                for seed_key, seed_data in eng_data.items():
                    if isinstance(seed_data, dict) and "ef_results" in seed_data:
                        for ef_result in seed_data["ef_results"]:
                            if ef_result["ef_search"] == ef:
                                row += f" {ef_result['recall_at_k']:.4f} |"
                                break
                        else:
                            row += " — |"
                        break
                else:
                    row += " — |"
            lines.append(row)

    elif table_name.startswith("latency"):
        # Latency table
        lines.append("| ef_search | " + " | ".join(
            f"{eng} p50 (ms)" for eng in engines
        ) + " | " + " | ".join(
            f"{eng} QPS (1T)" for eng in engines
        ) + " |")
        lines.append("| ---: | " + " | ".join("---:" for _ in engines) + " | " + " | ".join("---:" for _ in engines) + " |")

        for ef in params.get("ef_values", []):
            row = f"| {ef} |"
            lat_parts = []
            qps_parts = []
            for eng_name in engines:
                eng_data = engines[eng_name]
                for seed_key, seed_data in eng_data.items():
                    if isinstance(seed_data, dict) and "ef_results" in seed_data:
                        for ef_result in seed_data["ef_results"]:
                            if ef_result["ef_search"] == ef:
                                lat = ef_result.get("latency", {})
                                lat_parts.append(f" {lat.get('p50', 0):.3f} |")
                                qps = lat.get("sequential_qps", 0)
                                qps_parts.append(f" {qps:,.0f} |")
                                break
                        else:
                            lat_parts.append(" — |")
                            qps_parts.append(" — |")
                        break
                else:
                    lat_parts.append(" — |")
                    qps_parts.append(" — |")
            row += "".join(lat_parts) + "".join(qps_parts)
            lines.append(row)

    elif table_name.startswith("build"):
        # Build performance table
        lines.append("| Engine | Build Time (s) | RSS Delta (MB) |")
        lines.append("| :--- | ---: | ---: |")
        for eng_name in engines:
            eng_data = engines[eng_name]
            for seed_key, seed_data in eng_data.items():
                if isinstance(seed_data, dict) and "build_time_s" in seed_data:
                    lines.append(
                        f"| {eng_name} | {seed_data['build_time_s']:.3f} | "
                        f"{seed_data.get('rss_delta_mb', 0):.1f} |"
                    )
                    break

    elif table_name.startswith("scorecard"):
        # Delegated to make_scorecard.py
        lines.append("*Scorecard rendered by make_scorecard.py*")

    else:
        lines.append(f"*Unknown table type: {table_name}*")

    return "\n".join(lines)


def render_file(filepath: str, raw_results: dict, check_only: bool = False) -> tuple[bool, list[str]]:
    """Render generated blocks in a single file. Returns (changed, messages)."""
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    lines = content.split("\n")
    new_lines = []
    messages = []
    changed = False
    i = 0
    in_block = False
    block_table = ""
    block_run_id = ""

    while i < len(lines):
        line = lines[i]

        begin_match = BEGIN_PATTERN.search(line)
        if begin_match and not in_block:
            block_table = begin_match.group(1)
            block_run_id = begin_match.group(2)
            in_block = True
            new_lines.append(line)
            i += 1

            # Skip old content until END marker
            while i < len(lines) and not END_PATTERN.search(lines[i]):
                i += 1

            # Generate new content
            if block_run_id in raw_results:
                rendered = format_recall_table(raw_results[block_run_id], block_table)
                new_lines.append(rendered)
                messages.append(f"  Rendered table={block_table} run_id={block_run_id}")
            else:
                new_lines.append(f"*ERROR: run_id '{block_run_id}' not found in results/raw/*")
                messages.append(f"  MISSING: table={block_table} run_id={block_run_id}")

            # Add the END marker
            if i < len(lines):
                new_lines.append(lines[i])
            in_block = False
            i += 1
            continue

        new_lines.append(line)
        i += 1

    new_content = "\n".join(new_lines)
    if new_content != content:
        changed = True
        if not check_only:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(new_content)
            messages.append(f"  Updated: {filepath}")

    return changed, messages


def main():
    parser = argparse.ArgumentParser(description="Render generated blocks in docs from raw JSON")
    parser.add_argument("--check", action="store_true", help="Dry run, report what would change")
    parser.add_argument("--file", type=str, help="Specific file to render")
    args = parser.parse_args()

    raw_results = load_raw_results()
    print(f"Loaded {len(raw_results)} raw result file(s)")

    if args.file:
        files = [args.file]
    else:
        files = []
        for pattern in ["docs/*.md", "*.md"]:
            files.extend(glob.glob(os.path.join(ROOT, pattern)))
        files = list(set(files))

    total_changed = 0
    for filepath in sorted(files):
        changed, messages = render_file(filepath, raw_results, check_only=args.check)
        if changed:
            total_changed += 1
        for msg in messages:
            print(msg)

    action = "Would update" if args.check else "Updated"
    print(f"\n{action} {total_changed} file(s)")
    return 0 if total_changed == 0 or not args.check else 1


if __name__ == "__main__":
    sys.exit(main())
