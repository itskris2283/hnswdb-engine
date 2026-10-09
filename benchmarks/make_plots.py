import os
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd
import matplotlib.pyplot as plt


RESULTS_DIR = os.path.join(project_root, "results")


def generate_plots(csv_path=None):
    if csv_path is None:
        csv_path = os.path.join(RESULTS_DIR, "latest.csv")

    if not os.path.exists(csv_path):
        print(f"CSV file not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    if df.empty:
        print("CSV data is empty")
        return

    datasets = df["dataset"].unique()
    engines = df["engine"].unique()
    colors = {"hnswdb": "#1f77b4", "faiss_hnsw": "#ff7f0e", "hnswlib": "#2ca02c"}
    markers = {"hnswdb": "o", "faiss_hnsw": "s", "hnswlib": "^"}

    for ds in datasets:
        ds_df = df[df["dataset"] == ds]
        clean_ds = str(ds).replace("-", "_").replace(".", "_")

        # 1. Recall vs QPS (Sequential & Multi-thread Batch)
        plt.figure(figsize=(10, 6))
        for eng in engines:
            sub = ds_df[ds_df["engine"] == eng].sort_values(by="recall_at_10")
            if not sub.empty:
                qps_col = "batch_qps_1th" if "batch_qps_1th" in sub.columns else ("sequential_qps" if "sequential_qps" in sub.columns else "qps_1th")
                if qps_col in sub.columns:
                    plt.plot(
                        sub["recall_at_10"],
                        sub[qps_col],
                        marker=markers.get(eng, "o"),
                        color=colors.get(eng, "gray"),
                        label=f"{eng} (1 thread)",
                        linewidth=2,
                        markersize=7,
                    )
                # Find multi-thread col
                mt_cols = [c for c in sub.columns if (c.startswith("batch_qps_") or c.startswith("qps_")) and not c.endswith("_min") and not c.endswith("_max") and "1th" not in c]
                if mt_cols:
                    mt_col = mt_cols[-1]
                    plt.plot(
                        sub["recall_at_10"],
                        sub[mt_col],
                        marker=markers.get(eng, "o"),
                        color=colors.get(eng, "gray"),
                        linestyle="--",
                        label=f"{eng} ({mt_col.replace('batch_qps_', '').replace('qps_', '')})",
                        linewidth=2,
                        markersize=7,
                    )

        plt.title(f"Recall@10 vs. Throughput (QPS) - {ds}", fontsize=14, fontweight="bold")
        plt.xlabel("Recall@10", fontsize=12)
        plt.ylabel("Queries Per Second (QPS)", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(fontsize=11)
        plt.tight_layout()
        plot_path = os.path.join(RESULTS_DIR, f"recall_vs_qps_{clean_ds}.png")
        plt.savefig(plot_path, dpi=200)
        plt.close()
        print(f"Saved: {plot_path}")

        # 2. Recall vs p99 Latency
        plt.figure(figsize=(10, 6))
        for eng in engines:
            sub = ds_df[ds_df["engine"] == eng].sort_values(by="recall_at_10")
            if not sub.empty and "latency_p99_ms" in sub.columns:
                plt.plot(
                    sub["recall_at_10"],
                    sub["latency_p99_ms"],
                    marker=markers.get(eng, "o"),
                    color=colors.get(eng, "gray"),
                    label=eng,
                    linewidth=2,
                    markersize=7,
                )

        plt.title(f"Recall@10 vs. p99 Latency (ms) - {ds}", fontsize=14, fontweight="bold")
        plt.xlabel("Recall@10", fontsize=12)
        plt.ylabel("p99 Latency (ms) [lower is better]", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(fontsize=11)
        plt.tight_layout()
        plot_path = os.path.join(RESULTS_DIR, f"recall_vs_latency_{clean_ds}.png")
        plt.savefig(plot_path, dpi=200)
        plt.close()
        print(f"Saved: {plot_path}")

    # 3. Build Time & Disk Size Comparison Bar Chart
    summary_df = df.groupby(["dataset", "engine"])[["build_time_s", "disk_mb"]].first().reset_index()
    if not summary_df.empty:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Build time
        pivot_build = summary_df.pivot(index="dataset", columns="engine", values="build_time_s")
        pivot_build.plot(kind="bar", ax=axes[0], colormap="tab10", width=0.8)
        axes[0].set_title("Index Build Time (s) [lower is better]", fontsize=12, fontweight="bold")
        axes[0].set_ylabel("Seconds")
        axes[0].grid(axis="y", linestyle="--", alpha=0.6)
        axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=15, ha="right")

        # Disk size
        pivot_disk = summary_df.pivot(index="dataset", columns="engine", values="disk_mb")
        pivot_disk.plot(kind="bar", ax=axes[1], colormap="tab10", width=0.8)
        axes[1].set_title("Serialized Index Size (MB) [lower is better]", fontsize=12, fontweight="bold")
        axes[1].set_ylabel("Megabytes")
        axes[1].grid(axis="y", linestyle="--", alpha=0.6)
        axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=15, ha="right")

        plt.tight_layout()
        summary_plot = os.path.join(RESULTS_DIR, "engine_build_and_size_comparison.png")
        plt.savefig(summary_plot, dpi=200)
        plt.close()
        print(f"Saved: {summary_plot}")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else None
    generate_plots(path)
