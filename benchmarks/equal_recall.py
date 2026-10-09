import numpy as np
import pandas as pd
from typing import List, Dict, Optional


def interpolate_equal_recall(records: List[Dict], targets: List[float] = [0.90, 0.95, 0.99], recall_col: str = "recall_at_10") -> List[Dict]:
    """
    Perform log-linear interpolation across ef_search runs for each (dataset, engine, M, ef_construction, thread)
    to find QPS and latency at fixed recall targets (e.g., 0.90, 0.95, 0.99).
    """
    df = pd.DataFrame(records)
    if df.empty or recall_col not in df.columns:
        return []

    group_cols = [c for c in ["dataset", "engine", "M", "ef_construction"] if c in df.columns]
    results = []

    for keys, group in df.groupby(group_cols):
        grp_sorted = group.sort_values(by=recall_col).drop_duplicates(subset=[recall_col])
        recalls = grp_sorted[recall_col].values
        if len(recalls) < 1:
            continue

        base_info = grp_sorted.iloc[0].to_dict()

        for target in targets:
            out_row = {k: base_info[k] for k in group_cols}
            out_row["target_recall"] = target
            out_row["actual_min_recall"] = float(np.min(recalls))
            out_row["actual_max_recall"] = float(np.max(recalls))

            if target > np.max(recalls):
                out_row["status"] = "unreachable"
                out_row["interpolated_p50_ms"] = None
                out_row["interpolated_sequential_qps"] = None
                for col in grp_sorted.columns:
                    if col.startswith("batch_qps_") or col.startswith("qps_"):
                        out_row[f"interpolated_{col}"] = None
                results.append(out_row)
                continue

            out_row["status"] = "achieved"
            if len(recalls) == 1 or target <= np.min(recalls):
                idx = 0
                out_row["interpolated_p50_ms"] = float(grp_sorted["latency_p50_ms"].iloc[0])
                if "sequential_qps" in grp_sorted.columns:
                    out_row["interpolated_sequential_qps"] = float(grp_sorted["sequential_qps"].iloc[0])
                for col in grp_sorted.columns:
                    if col.startswith("batch_qps_") or col.startswith("qps_"):
                        out_row[f"interpolated_{col}"] = float(grp_sorted[col].iloc[0])
                results.append(out_row)
                continue

            # Find bounding segment
            upper_idx = np.searchsorted(recalls, target)
            lower_idx = max(0, upper_idx - 1)
            upper_idx = min(len(recalls) - 1, upper_idx)

            r1, r2 = recalls[lower_idx], recalls[upper_idx]
            if abs(r2 - r1) < 1e-9:
                alpha = 0.0
            else:
                alpha = (target - r1) / (r2 - r1)

            # Interpolate log-latency
            l1 = max(1e-6, float(grp_sorted["latency_p50_ms"].iloc[lower_idx]))
            l2 = max(1e-6, float(grp_sorted["latency_p50_ms"].iloc[upper_idx]))
            out_row["interpolated_p50_ms"] = round(float(np.exp(np.log(l1) + alpha * (np.log(l2) - np.log(l1)))), 4)

            # Interpolate QPS
            if "sequential_qps" in grp_sorted.columns:
                q1 = max(1.0, float(grp_sorted["sequential_qps"].iloc[lower_idx]))
                q2 = max(1.0, float(grp_sorted["sequential_qps"].iloc[upper_idx]))
                out_row["interpolated_sequential_qps"] = round(float(np.exp(np.log(q1) + alpha * (np.log(q2) - np.log(q1)))), 1)

            for col in grp_sorted.columns:
                if col.startswith("batch_qps_") or col.startswith("qps_"):
                    v1 = max(1.0, float(grp_sorted[col].iloc[lower_idx]))
                    v2 = max(1.0, float(grp_sorted[col].iloc[upper_idx]))
                    out_row[f"interpolated_{col}"] = round(float(np.exp(np.log(v1) + alpha * (np.log(v2) - np.log(v1)))), 1)

            results.append(out_row)

    return results


if __name__ == "__main__":
    import json
    import sys
    if len(sys.argv) > 1:
        with open(sys.argv[1]) as f:
            data = json.load(f)
        interp = interpolate_equal_recall(data)
        print(json.dumps(interp, indent=2))
