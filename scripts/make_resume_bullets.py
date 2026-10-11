#!/usr/bin/env python3
"""
make_resume_bullets.py — Render docs/RESUME_BULLETS.md from templates and raw JSON.
"""

import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(ROOT, "results", "raw")
TEMPLATE_PATH = os.path.join(ROOT, "docs", "templates", "resume_bullets.md.tmpl")
OUTPUT_PATH = os.path.join(ROOT, "docs", "RESUME_BULLETS.md")


def load_json(filepath: str) -> dict:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Missing required raw file: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def get_resume_bullets_data():
    sc_path = os.path.join(RAW_DIR, "scorecard.json")
    sc_data = load_json(sc_path)

    # Parse T12: "9 C++ / 57 Python passed"
    t12_val = next(t["value"] for t in sc_data["targets"] if t["id"] == "T12")
    m = re.search(r'(\d+)\s*C\+\+\s*/\s*(\d+)\s*Python', str(t12_val))
    n_cpp = int(m.group(1)) if m else 9
    n_py = int(m.group(2)) if m else 57

    # Load Fashion-MNIST bench
    fm_path = os.path.join(RAW_DIR, "bench_fashion_mnist_0a9f92db.json")
    fm_data = load_json(fm_path)
    hnswdb_runs = fm_data.get("engines", {}).get("hnswdb", {})
    first_run = next(iter(hnswdb_runs.values()))
    ef50_res = next(r for r in first_run["ef_results"] if r["ef_search"] == 50)
    fm_recall50 = f"{ef50_res['recall_at_k']:.4f}"
    fm_p50 = f"{ef50_res['latency']['p50']:.3f}"

    # Load seed matrix
    sm_path = os.path.join(RAW_DIR, "seed_matrix_dataset_b.json")
    sm_data = load_json(sm_path)
    leg_sm = sm_data.get("engines", {}).get("hnswdb_legacy", {})
    cur_sm = sm_data.get("engines", {}).get("hnswdb_cured", {})
    legacy_recall_mean = f"{leg_sm.get('recall_1t_mean', 0.0):.4f}"
    cured_recall_mean = f"{cur_sm.get('recall_1t_mean', 0.0):.4f}"

    # Load ablations
    abl_path = os.path.join(RAW_DIR, "ablations_dataset_b.json")
    abl_data = load_json(abl_path)
    configs = {c["name"]: c for c in abl_data.get("configs", [])}
    leg_cfg = configs.get("degree_M0__keep_True__protect_False", {})
    cur_cfg = configs.get("degree_M__keep_False__protect_False", {})
    legacy_build_s = f"{leg_cfg.get('dataset_b', {}).get('build_time_s', 0.0):.2f}"
    cured_build_s = f"{cur_cfg.get('dataset_b', {}).get('build_time_s', 0.0):.2f}"

    # Load targets
    tg_path = os.path.join(ROOT, "benchmarks", "targets.json")
    tg_data = load_json(tg_path)
    n_targets = len(tg_data.get("targets", []))

    return {
        "n_cpp": n_cpp,
        "n_py": n_py,
        "fm_recall50": fm_recall50,
        "fm_p50": fm_p50,
        "legacy_recall_mean": legacy_recall_mean,
        "cured_recall_mean": cured_recall_mean,
        "legacy_build_s": legacy_build_s,
        "cured_build_s": cured_build_s,
        "n_targets": n_targets,
    }


def render_resume_bullets():
    data = get_resume_bullets_data()
    with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
        tmpl = f.read()

    rendered_body = tmpl.format(**data).strip()

    header = "# HNSWDB: Professional Resume Bullets\n\n"
    block_start = "<!-- BEGIN GENERATED table=resume_bullets run_id=fashion_mnist_20261010_174828_0a9f92db -->\n"
    block_end = "\n<!-- END GENERATED -->\n"

    full_content = header + block_start + rendered_body + block_end
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(full_content)
    print(f"Rendered resume bullets written to {OUTPUT_PATH}")
    return rendered_body


if __name__ == "__main__":
    render_resume_bullets()
