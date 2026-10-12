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
import re
import hashlib
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(ROOT, "results", "raw")
MANIFEST_PATH = os.path.join(RAW_DIR, "MANIFEST.json")


def compare_self_retrieval(
    h_unreach: float,
    h_denom: int,
    h_builds: int,
    b_unreach: float,
    b_denom: int,
    b_builds: int,
    b_eng: str,
    workload: str,
    source_str: str,
) -> dict | None:
    """
    Compare self-retrieval failure rates between HNSWDB and a baseline.
    Raises ValueError if denominators differ.
    Returns comparison dict if HNSWDB failure rate is worse than baseline by > 5% relative, else None.
    """
    if h_denom != b_denom:
        raise ValueError(
            f"Mismatched denominators in self-retrieval comparison: "
            f"HNSWDB denominator={h_denom}, baseline ({b_eng}) denominator={b_denom}"
        )
    h_fail_rate = h_unreach / h_denom
    b_fail_rate = b_unreach / b_denom

    # HNSWDB loses if failure rate is > baseline failure rate * 1.05
    if b_fail_rate > 0 and h_fail_rate > b_fail_rate * 1.05:
        gap_rel = ((h_fail_rate - b_fail_rate) / b_fail_rate) * 100.0
        ratio = h_fail_rate / b_fail_rate
        return {
            "workload": workload,
            "metric": f"Self-retrieval failure rate (count / {h_denom:,}, n={h_builds} vs n={b_builds})",
            "hnswdb": f"{h_unreach:.2f} / {h_denom:,} ({h_fail_rate*100:.3f}%, n={h_builds})",
            "best_baseline": f"{b_unreach:.2f} / {b_denom:,} ({b_fail_rate*100:.3f}%, {b_eng}, n={b_builds})",
            "gap": f"+{gap_rel:.1f}% ({ratio:.2f}x)",
            "source": source_str,
        }
    return None


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
    bsr_files = [f for f in sorted(os.listdir(RAW_DIR)) if f.startswith("consistent_self_retrieval_") and f.endswith(".json")]
    if not bsr_files:
        bsr_files = [f for f in sorted(os.listdir(RAW_DIR)) if f.startswith("baseline_self_retrieval_") and f.endswith(".json")]
    bsr_data = None
    bsr_fname = ""
    if bsr_files:
        bsr_fname = bsr_files[-1]
        with open(os.path.join(RAW_DIR, bsr_fname), "r", encoding="utf-8") as bf:
            bsr_data = json.load(bf)

    # Dataset B mean self-retrieval failures
    sm_path = os.path.join(RAW_DIR, "seed_matrix_dataset_b.json")
    if os.path.exists(sm_path) and bsr_data:
        with open(sm_path, "r", encoding="utf-8") as f:
            sm = json.load(f)
        cur_unreach = sm.get("engines", {}).get("hnswdb_cured", {}).get("unreachable_mean", 9.32)
        h_denom = 20000
        h_builds = 25
        db_bsr = bsr_data.get("dataset_b", {})
        hl_info = db_bsr.get("hnswlib", {})
        fa_info = db_bsr.get("faiss", {})

        candidates = []
        if "unreachable_mean" in hl_info:
            candidates.append((hl_info["unreachable_mean"], hl_info.get("denominator", h_denom), hl_info.get("n_builds", 5), "hnswlib", f"{bsr_fname}:dataset_b.hnswlib.unreachable_mean"))
        if "unreachable_mean" in fa_info:
            candidates.append((fa_info["unreachable_mean"], fa_info.get("denominator", h_denom), fa_info.get("n_builds", 5), "FAISS", f"{bsr_fname}:dataset_b.faiss.unreachable_mean"))

        if candidates:
            # Best baseline has lowest failure rate
            best_b = min(candidates, key=lambda c: c[0] / c[1])
            comp = compare_self_retrieval(
                h_unreach=cur_unreach,
                h_denom=h_denom,
                h_builds=h_builds,
                b_unreach=best_b[0],
                b_denom=best_b[1],
                b_builds=best_b[2],
                b_eng=best_b[3],
                workload="Dataset B (20K x 64d, sigma=0.1)",
                source_str=f"seed_matrix_dataset_b.json:engines[hnswdb_cured].unreachable_mean; {best_b[4]}",
            )
            if comp is not None:
                comparisons.append(comp)

    # Fashion-MNIST self-retrieval failures
    if bsr_data:
        fm_bsr = bsr_data.get("fashion_mnist", {})
        fm_full = fm_bsr.get("full_index_60000", fm_bsr)
        h_info = fm_full.get("hnswdb", {})
        hl_info = fm_full.get("hnswlib", {})
        fa_info = fm_full.get("faiss", {})

        if "unreachable_count" in h_info:
            h_unreach = h_info["unreachable_count"]
            h_denom = h_info.get("denominator", 60000)
            h_builds = h_info.get("n_builds", 1)

            candidates = []
            if "unreachable_count" in hl_info:
                candidates.append((hl_info["unreachable_count"], hl_info.get("denominator", h_denom), hl_info.get("n_builds", 1), "hnswlib", f"{bsr_fname}:fashion_mnist.full_index_60000.hnswlib.unreachable_count"))
            if "unreachable_count" in fa_info:
                candidates.append((fa_info["unreachable_count"], fa_info.get("denominator", h_denom), fa_info.get("n_builds", 1), "FAISS", f"{bsr_fname}:fashion_mnist.full_index_60000.faiss.unreachable_count"))

            if candidates:
                best_b = min(candidates, key=lambda c: c[0] / c[1])
                comp = compare_self_retrieval(
                    h_unreach=h_unreach,
                    h_denom=h_denom,
                    h_builds=h_builds,
                    b_unreach=best_b[0],
                    b_denom=best_b[1],
                    b_builds=best_b[2],
                    b_eng=best_b[3],
                    workload="Fashion-MNIST (full index, 60,000 nodes)",
                    source_str=f"{bsr_fname}:fashion_mnist.full_index_60000.hnswdb.unreachable_count; {best_b[4]}",
                )
                if comp is not None:
                    comparisons.append(comp)

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

    # Update marker in README.md
    readme_path = os.path.join(ROOT, "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as rf:
            r_text = rf.read()
        new_r_text = re.sub(
            r"<!-- BEGIN GENERATED table=where_it_loses run_id=[^\s>]+ -->",
            f"<!-- BEGIN GENERATED table=where_it_loses run_id={run_id} -->",
            r_text
        )
        if new_r_text != r_text:
            with open(readme_path, "w", encoding="utf-8") as rf:
                rf.write(new_r_text)
            print(f"Updated README.md marker with run_id={run_id}")

    return run_id


def main():
    generate_and_save()


if __name__ == "__main__":
    main()
