#!/usr/bin/env python3
"""
where_it_loses.py — Systematically identify all benchmarks where HNSWDB
is worse than the best reference baseline by more than 5 percent.

Inputs (strictly limited to):
- equal_recall_grid.json
- bench_A_a9f1a6f6.json, bench_B_s01_8a2302be.json, bench_fashion_mnist_0a9f92db.json
- multithread_build.json
- seed_matrix_dataset_b.json
- self-retrieval files
"""

import json
import os
import hashlib
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(ROOT, "results", "raw")
MANIFEST_PATH = os.path.join(RAW_DIR, "MANIFEST.json")


def analyze_losing_comparisons() -> tuple[list[dict], str]:
    comparisons = []

    # 1. Equal recall sequential QPS, 4T QPS, 8T QPS at 0.95 and 0.99
    eq_path = os.path.join(RAW_DIR, "equal_recall_grid.json")
    if os.path.exists(eq_path):
        with open(eq_path, "r", encoding="utf-8") as f:
            eq = json.load(f)

        ds_names = {
            "A": "Dataset A (5K x 32d)",
            "B_s01": "Dataset B (20K x 64d, sigma=0.1)",
            "fashion_mnist": "Fashion-MNIST (60K x 784d)",
        }

        eq_map = {}
        for r in eq.get("interpolations", []):
            key = (r["dataset"], r["target_recall"])
            if key not in eq_map:
                eq_map[key] = {}
            eq_map[key][r["engine"]] = r

        for ds_code in ["A", "B_s01", "fashion_mnist"]:
            for tgt in [0.95, 0.99]:
                key = (ds_code, tgt)
                if key not in eq_map:
                    continue
                engs = eq_map[key]
                if "hnswdb" not in engs:
                    continue
                h_row = engs["hnswdb"]

                for metric_key, metric_label in [
                    ("interpolated_sequential_qps", "Sequential QPS (equal recall)"),
                    ("interpolated_batch_qps_4th", "4-thread Batch QPS (equal recall)"),
                    ("interpolated_batch_qps_8th", "8-thread Batch QPS (equal recall)"),
                ]:
                    h_val = h_row.get(metric_key, 0.0)
                    best_b = 0.0
                    best_eng = ""
                    for eng in ["faiss", "hnswlib"]:
                        if eng in engs:
                            b_val = engs[eng].get(metric_key, 0.0)
                            if b_val > best_b:
                                best_b = b_val
                                best_eng = eng

                    # Lower QPS is worse
                    if best_b > 0 and h_val < best_b * 0.95:
                        gap_pct = ((best_b - h_val) / best_b) * 100.0
                        comparisons.append({
                            "workload": f"{ds_names[ds_code]} @ R={tgt:.2f}",
                            "metric": metric_label,
                            "hnswdb": f"{h_val:.1f} QPS",
                            "best_baseline": f"{best_b:.1f} QPS ({best_eng})",
                            "gap": f"-{gap_pct:.1f}%",
                            "source": f"equal_recall_grid.json:interpolations[dataset={ds_code},target_recall={tgt}].{metric_key}",
                        })

    # 2. Single-thread build time, RSS delta, and ef-recall from bench files
    bench_specs = [
        ("A", "Dataset A (5K x 32d)", "bench_A_a9f1a6f6.json"),
        ("B_s01", "Dataset B (20K x 64d, sigma=0.1)", "bench_B_s01_8a2302be.json"),
        ("fashion_mnist", "Fashion-MNIST (60K x 784d)", "bench_fashion_mnist_0a9f92db.json"),
    ]

    for ds_code, ds_name, fname in bench_specs:
        fpath = os.path.join(RAW_DIR, fname)
        if not os.path.exists(fpath):
            continue
        with open(fpath, "r", encoding="utf-8") as f:
            b_data = json.load(f)

        engs = b_data.get("engines", {})
        h_s42 = engs.get("hnswdb", {}).get("seed_42", {})
        fa_s42 = engs.get("faiss", {}).get("seed_42", {})
        hl_s42 = engs.get("hnswlib", {}).get("seed_42", {})

        # Single-thread build time (higher is worse)
        h_bld = h_s42.get("build_time_s", 0.0)
        best_bld = float("inf")
        best_bld_eng = ""
        for eng, s42 in [("FAISS", fa_s42), ("hnswlib", hl_s42)]:
            b_bld = s42.get("build_time_s", float("inf"))
            if b_bld < best_bld:
                best_bld = b_bld
                best_bld_eng = eng
        if best_bld < float("inf") and h_bld > best_bld * 1.05:
            gap_pct = ((h_bld - best_bld) / best_bld) * 100.0
            comparisons.append({
                "workload": ds_name,
                "metric": "Single-thread build time",
                "hnswdb": f"{h_bld:.4f} s" if len(str(h_bld).split(".")[-1]) > 3 else f"{h_bld:.3f} s",
                "best_baseline": f"{best_bld:.4f} s ({best_bld_eng})" if len(str(best_bld).split(".")[-1]) > 3 else f"{best_bld:.3f} s ({best_bld_eng})",
                "gap": f"+{gap_pct:.1f}%",
                "source": f"{fname}:engines[hnswdb].seed_42.build_time_s",
            })

        # Process RSS delta (higher is worse)
        h_rss = h_s42.get("rss_delta_mb", 0.0)
        best_rss = float("inf")
        best_rss_eng = ""
        for eng, s42 in [("FAISS", fa_s42), ("hnswlib", hl_s42)]:
            b_rss = s42.get("rss_delta_mb", float("inf"))
            if b_rss < best_rss:
                best_rss = b_rss
                best_rss_eng = eng
        if best_rss < float("inf") and h_rss > best_rss * 1.05:
            gap_pct = ((h_rss - best_rss) / best_rss) * 100.0
            comparisons.append({
                "workload": ds_name,
                "metric": "Process RSS delta",
                "hnswdb": f"{h_rss:.2f} MB",
                "best_baseline": f"{best_rss:.2f} MB ({best_rss_eng})",
                "gap": f"+{gap_pct:.1f}%",
                "source": f"{fname}:engines[hnswdb].seed_42.rss_delta_mb",
            })

        # ef-search recall (> 0.005 lower)
        h_ef = {r["ef_search"]: r["recall_at_k"] for r in h_s42.get("ef_results", [])}
        fa_ef = {r["ef_search"]: r["recall_at_k"] for r in fa_s42.get("ef_results", [])}
        hl_ef = {r["ef_search"]: r["recall_at_k"] for r in hl_s42.get("ef_results", [])}
        for ef_val in sorted(h_ef.keys()):
            hr = h_ef[ef_val]
            best_rec = 0.0
            best_rec_eng = ""
            for eng, ef_dict in [("FAISS", fa_ef), ("hnswlib", hl_ef)]:
                if ef_dict.get(ef_val, 0.0) > best_rec:
                    best_rec = ef_dict.get(ef_val, 0.0)
                    best_rec_eng = eng
            if best_rec - hr > 0.005:
                gap_abs = best_rec - hr
                comparisons.append({
                    "workload": f"{ds_name} (ef_search={ef_val})",
                    "metric": "Recall@10",
                    "hnswdb": f"{hr:.4f}",
                    "best_baseline": f"{best_rec:.4f} ({best_rec_eng})",
                    "gap": f"-{gap_abs:.4f} abs",
                    "source": f"{fname}:engines[hnswdb].seed_42.ef_results[ef={ef_val}].recall_at_k",
                })

    # 3. 4-thread build time from multithread_build.json
    mb_path = os.path.join(RAW_DIR, "multithread_build.json")
    if os.path.exists(mb_path):
        with open(mb_path, "r", encoding="utf-8") as f:
            mb = json.load(f)
        for ds_key, ds_name in [
            ("dataset_b", "Dataset B (20K x 64d, sigma=0.1)"),
            ("fashion_mnist", "Fashion-MNIST (60K x 784d)"),
        ]:
            d_mb = mb.get(ds_key, {})
            h_bld = d_mb.get("hnswdb_build_s", 0.0)
            hl_bld = d_mb.get("hnswlib_build_s", 0.0)
            if hl_bld > 0 and h_bld > hl_bld * 1.05:
                gap_pct = ((h_bld - hl_bld) / hl_bld) * 100.0
                comparisons.append({
                    "workload": ds_name,
                    "metric": "4-thread build time",
                    "hnswdb": f"{h_bld:.3f} s",
                    "best_baseline": f"{hl_bld:.3f} s (hnswlib)",
                    "gap": f"+{gap_pct:.1f}%",
                    "source": f"multithread_build.json:{ds_key}.hnswdb_build_s",
                })

    # 4. Self-retrieval comparisons (Dataset B seed matrix and Fashion-MNIST self-retrieval)
    bsr_files = [f for f in os.listdir(RAW_DIR) if f.startswith("baseline_self_retrieval_") and f.endswith(".json")]
    bsr_data = None
    bsr_fname = ""
    if bsr_files:
        bsr_fname = bsr_files[0]
        with open(os.path.join(RAW_DIR, bsr_fname), "r", encoding="utf-8") as bf:
            bsr_data = json.load(bf)

    # Dataset B mean self-retrieval failures
    sm_path = os.path.join(RAW_DIR, "seed_matrix_dataset_b.json")
    if os.path.exists(sm_path) and bsr_data:
        with open(sm_path, "r", encoding="utf-8") as f:
            sm = json.load(f)
        cur_unreach = sm.get("engines", {}).get("hnswdb_cured", {}).get("unreachable_mean")
        db_bsr = bsr_data.get("dataset_b", {})
        hl_unreach = db_bsr.get("hnswlib", {}).get("unreachable_mean")
        fa_unreach = db_bsr.get("faiss", {}).get("unreachable_mean")

        valid_b = []
        if hl_unreach is not None:
            valid_b.append((hl_unreach, "hnswlib", f"{bsr_fname}:dataset_b.hnswlib.unreachable_mean"))
        if fa_unreach is not None:
            valid_b.append((fa_unreach, "FAISS", f"{bsr_fname}:dataset_b.faiss.unreachable_mean"))

        if cur_unreach is not None and valid_b:
            best_b_val, best_b_eng, best_b_src = min(valid_b, key=lambda x: x[0])
            if cur_unreach > best_b_val * 1.05:
                gap_abs = cur_unreach - best_b_val
                ratio = cur_unreach / best_b_val if best_b_val > 0 else float("inf")
                comparisons.append({
                    "workload": "Dataset B (20K x 64d, sigma=0.1)",
                    "metric": "Unreachable nodes (self-retrieval, mean over n builds)",
                    "hnswdb": f"{cur_unreach:.2f} (n=25)",
                    "best_baseline": f"{best_b_val:.1f} ({best_b_eng}, n=5)",
                    "gap": f"+{gap_abs:.2f} ({ratio:.2f}x)",
                    "source": f"seed_matrix_dataset_b.json:engines[hnswdb_cured].unreachable_mean; {best_b_src}",
                })

    # Fashion-MNIST self-retrieval failures
    fmn_path = os.path.join(RAW_DIR, "self_retrieval_fashion_mnist_50033fba.json")
    if os.path.exists(fmn_path) and bsr_data:
        with open(fmn_path, "r", encoding="utf-8") as f:
            fmn_data = json.load(f)
        h_unreach = fmn_data.get("metrics", {}).get("unreachable_nodes")
        fm_bsr = bsr_data.get("fashion_mnist", {})
        hl_unreach = fm_bsr.get("hnswlib", {}).get("unreachable_count")
        fa_unreach = fm_bsr.get("faiss", {}).get("unreachable_count")

        valid_b = []
        if hl_unreach is not None:
            valid_b.append((hl_unreach, "hnswlib", f"{bsr_fname}:fashion_mnist.hnswlib.unreachable_count"))
        elif "hnswlib" in fm_bsr:
            valid_b.append((None, "hnswlib (not measured)", ""))
        if fa_unreach is not None:
            valid_b.append((fa_unreach, "FAISS", f"{bsr_fname}:fashion_mnist.faiss.unreachable_count"))
        elif "faiss" in fm_bsr:
            valid_b.append((None, "FAISS (not measured)", ""))

        measured_b = [b for b in valid_b if b[0] is not None]
        if h_unreach is not None and measured_b:
            best_b_val, best_b_eng, best_b_src = min(measured_b, key=lambda x: x[0])
            if h_unreach > best_b_val * 1.05:
                gap_abs = h_unreach - best_b_val
                ratio = h_unreach / best_b_val if best_b_val > 0 else float("inf")
                comparisons.append({
                    "workload": "Fashion-MNIST (60K x 784d)",
                    "metric": "Unreachable nodes (self-retrieval, mean over n builds)",
                    "hnswdb": f"{h_unreach} (n=1)",
                    "best_baseline": f"{best_b_val} ({best_b_eng}, n=1)",
                    "gap": f"+{int(gap_abs)} ({ratio:.2f}x)",
                    "source": f"self_retrieval_fashion_mnist_50033fba.json:metrics.unreachable_nodes; {best_b_src}",
                })

    # Deterministic content hash for run_id
    comp_json = json.dumps(comparisons, sort_keys=True, indent=2)
    content_hash = hashlib.sha256(comp_json.encode("utf-8")).hexdigest()[:8]
    run_id = f"where_it_loses_{content_hash}"

    return comparisons, run_id


def generate_and_save():
    comparisons, run_id = analyze_losing_comparisons()
    output_data = {
        "run_id": run_id,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "threshold_pct": 5.0,
        "comparisons": comparisons,
    }

    out_file = os.path.join(RAW_DIR, "where_it_loses.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
    print(f"Saved: {out_file} (run_id={run_id}, {len(comparisons)} comparisons found)")

    # Update manifest
    with open(out_file, "rb") as f:
        b = f.read()
    sha = hashlib.sha256(b).hexdigest()
    sz = len(b)

    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        runs = manifest.get("runs", [])
        found = False
        for r in runs:
            if r.get("file") == "where_it_loses.json":
                r["run_id"] = run_id
                r["sha256"] = sha
                r["size_bytes"] = sz
                found = True
                break
        if not found:
            runs.append({
                "file": "where_it_loses.json",
                "run_id": run_id,
                "dataset": "cross_dataset",
                "label": "Where HNSWDB Loses (>5% Gap)",
                "sha256": sha,
                "size_bytes": sz,
            })
        manifest["runs"] = runs
        with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        print("Updated MANIFEST.json with where_it_loses.json")

    return run_id


def main():
    generate_and_save()


if __name__ == "__main__":
    main()
