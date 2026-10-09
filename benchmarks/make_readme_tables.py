import os
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd


RESULTS_DIR = os.path.join(project_root, "results")


def generate_tables(csv_path=None):
    if csv_path is None:
        csv_path = os.path.join(RESULTS_DIR, "latest.csv")

    if not os.path.exists(csv_path):
        return "No benchmark results found."

    df = pd.read_csv(csv_path)
    if df.empty:
        return "Empty benchmark results."

    output_lines = []
    output_lines.append("## Benchmark Summary Results\n")

    datasets = df["dataset"].unique()
    for ds in datasets:
        ds_df = df[df["dataset"] == ds]
        output_lines.append(f"### Dataset: `{ds}`\n")

        # Table: Recall, Latency, QPS
        cols = ["engine", "ef_search", "recall_at_10", "approx_ratio", "latency_p50_ms", "latency_p99_ms", "sequential_qps"]
        if "batch_qps_1th" in ds_df.columns:
            cols.append("batch_qps_1th")
        if "batch_qps_4th" in ds_df.columns:
            cols.append("batch_qps_4th")

        avail_cols = [c for c in cols if c in ds_df.columns]
        sub = ds_df[avail_cols].copy()
        sub = sub.sort_values(by=["engine", "ef_search"])
        output_lines.append(sub.to_markdown(index=False))
        output_lines.append("\n")

        # Table: Build Time, RSS delta & Disk Size
        b_cols = ["engine", "build_time_s", "disk_mb"]
        if "rss_delta_mb" in ds_df.columns:
            b_cols.append("rss_delta_mb")
        avail_b_cols = [c for c in b_cols if c in ds_df.columns]
        build_df = ds_df.groupby("engine")[avail_b_cols[1:]].first().reset_index()
        output_lines.append("#### Build Time, Memory RSS & Storage Footprint\n")
        output_lines.append(build_df.to_markdown(index=False))
        output_lines.append("\n---\n")

    table_md = "\n".join(output_lines)
    md_path = os.path.join(RESULTS_DIR, "tables.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(table_md)
    print(f"Generated Markdown tables at: {md_path}")
    return table_md


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else None
    print(generate_tables(path))
