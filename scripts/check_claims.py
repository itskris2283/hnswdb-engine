#!/usr/bin/env python3
r"""
Strict Claims and Consistency Checker for HNSWDB (Round 5).

Enforces:
1. Banned phrases scanned across README.md, REPORT.md, docs/*.md with file:line output.
2. No numbers with units (ms, us, ns, s, QPS, MB, GB, x, %) or recall decimals (0.xxxx)
   outside generated blocks, unless covered by a strictly scoped allowlist entry.
3. Allowlist format: <= 25 entries of '<file>:<line-regex> | <value> | <reason>'.
   No bare small integers. No threshold values from benchmarks/targets.json.
   No allowlisted value matching a metric in results/raw/*.json.
4. Protected phrase rules:
   - 'root cause' and 'hypothesis confirmed' allowed only in docs/INVESTIGATIONS.md
     and only if immediately followed by an 'evidence: <path>' line to an existing file.
   - 'L2SqrSIMD16Ext' allowed only on a line stating it is hnswlib's with no performance claim.
   - 'sub-millisecond' allowed only inside a generated block naming ef and percentile.
5. Absolute paths forbidden.
6. Legacy .hnsw extension forbidden.
"""

import os
import re
import sys
import json
import glob

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS_DIR = os.path.join(PROJECT_ROOT, "docs")
README_PATH = os.path.join(PROJECT_ROOT, "README.md")
REPORT_PATH = os.path.join(PROJECT_ROOT, "REPORT.md")
ALLOWLIST_PATH = os.path.join(PROJECT_ROOT, "docs", "claims_allowlist.txt")
OLD_WHITELIST_PATH = os.path.join(PROJECT_ROOT, "docs", "claims_whitelist.txt")
RAW_RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "raw")
TARGETS_PATH = os.path.join(PROJECT_ROOT, "benchmarks", "targets.json")

BEGIN_GENERATED = re.compile(r'<!--\s*BEGIN\s+GENERATED\s+')
END_GENERATED = re.compile(r'<!--\s*END\s+GENERATED\s*-->')

BANNED_PHRASES = [
    "production persistence",
    "production-grade",
    "lock-free",
    "real algorithmic limitation",
    "pathological",
    "surpass",
    "outperform",
    "beats faiss",
    "beats hnswlib",
    "faster than hnswlib",
    "tsan-clean",
    "threadsanitizer-clean",
    "under 0.3 ms",
    "edge starvation",
    "cutting incoming edges",
    "pruned-element retention",
    "instruction profiling",
    "distance compute dominates",
    "reduced unreachable nodes to",
    "scheduling race conditions",
    "real algorithmic",
]

# Regex for numbers with units: ms, us, µs, ns, s, QPS, MB, GB, x, %
NUMBER_WITH_UNIT_RE = re.compile(
    r'(?<![a-zA-Z0-9_])(\d+(?:\.\d+)?\s*(?:ms|us|µs|ns|s|QPS|MB|GB|x|%))(?![a-zA-Z0-9_])',
    re.IGNORECASE
)

# Recall-style decimals (e.g. 0.9991, 0.9716, 0.0284)
RECALL_DECIMAL_RE = re.compile(r'(?<![a-zA-Z0-9_.])(0\.\d{4})(?![a-zA-Z0-9_.])')


def load_raw_metric_values(raw_dir=RAW_RESULTS_DIR):
    """Collect all numeric scalar metric values from raw JSON files."""
    values = set()
    if not os.path.exists(raw_dir):
        return values

    for jpath in glob.glob(os.path.join(raw_dir, "*.json")):
        if os.path.basename(jpath) == "MANIFEST.json":
            continue
        try:
            with open(jpath, "r", encoding="utf-8") as f:
                data = json.load(f)

            def walk(obj):
                if isinstance(obj, (int, float)) and not isinstance(obj, bool):
                    values.add(str(obj))
                    values.add(f"{float(obj):.4f}")
                    values.add(f"{float(obj):.3f}")
                    values.add(f"{float(obj):.2f}")
                    values.add(f"{float(obj):.1f}")
                elif isinstance(obj, dict):
                    for v in obj.values():
                        walk(v)
                elif isinstance(obj, list):
                    for item in obj:
                        walk(item)

            walk(data)
        except Exception:
            pass
    return values


def load_target_thresholds(targets_path=TARGETS_PATH):
    """Load threshold values from targets.json if present."""
    thresholds = set()
    if os.path.exists(targets_path):
        try:
            with open(targets_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            targets = data.get("targets", []) if isinstance(data, dict) else data
            for t in targets:
                if isinstance(t, dict) and "threshold" in t:
                    thresholds.add(str(t["threshold"]))
                    if isinstance(t["threshold"], (int, float)):
                        thresholds.add(f"{float(t['threshold']):.4f}")
                        thresholds.add(f"{float(t['threshold']):.3f}")
                        thresholds.add(f"{float(t['threshold']):.2f}")
        except Exception:
            pass
    return thresholds


def parse_allowlist(allowlist_path=ALLOWLIST_PATH, raw_metric_vals=None, target_thresholds=None):
    """
    Parses and strictly validates the allowlist.
    Returns: list of (file, line_regex, value, reason), list of errors.
    """
    entries = []
    errors = []

    if not os.path.exists(allowlist_path):
        return entries, errors

    with open(allowlist_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    raw_metrics = raw_metric_vals if raw_metric_vals is not None else load_raw_metric_values()
    thresholds = target_thresholds if target_thresholds is not None else load_target_thresholds()

    entry_count = 0
    for idx, raw_line in enumerate(lines, 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        entry_count += 1
        parts = [p.strip() for p in line.split("|")]
        if len(parts) != 3:
            errors.append(
                f"{allowlist_path}:{idx}: Malformed entry (expected '<file>:<line-regex> | <value> | <reason>'): {line}"
            )
            continue

        scope, value, reason = parts
        if ":" not in scope:
            errors.append(f"{allowlist_path}:{idx}: Unscoped allowlist entry (missing '<file>:<regex>'): {line}")
            continue

        target_file, line_regex = scope.split(":", 1)
        target_file = target_file.strip().replace("\\", "/")
        line_regex = line_regex.strip()

        if not target_file or not line_regex:
            errors.append(f"{allowlist_path}:{idx}: Both file and line-regex must be non-empty: {line}")
            continue

        # Check for bare small integers
        if re.fullmatch(r'\d+', value):
            errors.append(
                f"{allowlist_path}:{idx}: Bare integer '{value}' forbidden in allowlist (bare integers are not numbers with units)"
            )
            continue

        # Check against target thresholds
        if value in thresholds:
            errors.append(
                f"{allowlist_path}:{idx}: Allowlist entry '{value}' matches fixed target threshold (forbidden)"
            )
            continue

        # Check against raw JSON metric values
        clean_num_match = re.search(r'(\d+(?:\.\d+)?)', value)
        if clean_num_match:
            num_str = clean_num_match.group(1)
            if num_str in raw_metrics:
                errors.append(
                    f"{allowlist_path}:{idx}: Value '{value}' matches raw JSON metric {num_str} (cannot allowlist measured metrics)"
                )
                continue

        # Verify that target_file exists
        full_target = os.path.join(PROJECT_ROOT, target_file)
        if not os.path.exists(full_target):
            errors.append(f"{allowlist_path}:{idx}: Target file '{target_file}' does not exist on disk")
            continue

        entries.append({
            "file": target_file,
            "regex": re.compile(line_regex),
            "value": value,
            "reason": reason,
            "line_num": idx,
        })

    if entry_count > 25:
        errors.append(f"{allowlist_path}: Allowlist exceeds 25-entry cap (found {entry_count} entries)")

    return entries, errors


def check_claims(root_dir=PROJECT_ROOT):
    errors = []

    # 1. Stale whitelist check
    old_wl = os.path.join(root_dir, "docs", "claims_whitelist.txt")
    if os.path.exists(old_wl):
        errors.append("docs/claims_whitelist.txt still exists — must be deleted")

    # 2. Parse and validate allowlist
    al_path = os.path.join(root_dir, "docs", "claims_allowlist.txt")
    allowlist_entries, al_errors = parse_allowlist(allowlist_path=al_path)
    errors.extend(al_errors)

    # 3. Discover files to check
    files_to_check = []
    readme = os.path.join(root_dir, "README.md")
    report = os.path.join(root_dir, "REPORT.md")
    if os.path.exists(readme):
        files_to_check.append(readme)
    if os.path.exists(report):
        files_to_check.append(report)

    docs_dir = os.path.join(root_dir, "docs")
    if os.path.exists(docs_dir):
        for fname in sorted(os.listdir(docs_dir)):
            if fname.endswith(".md"):
                files_to_check.append(os.path.join(docs_dir, fname))

    for fpath in files_to_check:
        rel_path = os.path.relpath(fpath, root_dir).replace("\\", "/")
        with open(fpath, "r", encoding="utf-8") as f:
            lines = f.readlines()

        in_generated = False
        in_corrections = False

        for line_idx, line in enumerate(lines, 1):
            stripped = line.strip()

            # Check generated block boundary
            if BEGIN_GENERATED.search(line):
                in_generated = True
                continue
            if END_GENERATED.search(line):
                in_generated = False
                continue

            # Check corrections block boundary in REPORT.md
            if rel_path == "REPORT.md" and "<!-- BEGIN CORRECTIONS -->" in line:
                in_corrections = True
                continue
            if rel_path == "REPORT.md" and "<!-- END CORRECTIONS -->" in line:
                in_corrections = False
                continue

            # Absolute path checks
            if "file:///" in line:
                errors.append(f"{rel_path}:{line_idx}: Prohibited 'file:///' link found")
            if re.search(r'[A-Za-z]:[\\/][A-Za-z0-9_-]', line):
                errors.append(f"{rel_path}:{line_idx}: Prohibited Windows absolute path found")

            # Legacy index.hnsw reference
            if re.search(r'(?<![a-zA-Z0-9_-])index\.hnsw(?![a-zA-Z0-9_-])', line):
                errors.append(f"{rel_path}:{line_idx}: Reference to legacy 'index.hnsw' instead of '.hdb'")

            # Banned phrases (exempt inside REPORT.md Corrections section where earlier claims are explicitly retracted)
            if not in_corrections:
                line_lower = line.lower()
                for phrase in BANNED_PHRASES:
                    if phrase in line_lower:
                        if phrase == "edge starvation" and "does not explain the gap" in line_lower:
                            continue
                        errors.append(f"{rel_path}:{line_idx}: Contains prohibited phrase '{phrase}'")

                if "race conditions" in line_lower:
                    approved_c11 = "parallel builds insert nodes in a nondeterministic order"
                    if approved_c11 not in line_lower:
                        errors.append(f"{rel_path}:{line_idx}: Contains prohibited phrase 'race conditions'")

                if "lock contention" in line_lower:
                    ev_match = re.search(r'(?:results/raw/|\.\./results/raw/)([a-zA-Z0-9_\-\.]+)', line)
                    has_ev = False
                    if ev_match:
                        raw_f = os.path.join(root_dir, "results", "raw", ev_match.group(1))
                        if os.path.exists(raw_f):
                            has_ev = True
                    if not has_ev:
                        errors.append(f"{rel_path}:{line_idx}: Contains prohibited phrase 'lock contention' without raw evidence path on the same line")

            # Special phrase: sub-millisecond
            if "sub-millisecond" in line_lower:
                has_ef = bool(re.search(r'\bef\b|\bef=', line, re.IGNORECASE))
                has_percentile = bool(re.search(r'\bp\d+\b', line, re.IGNORECASE))
                if not (in_generated and has_ef and has_percentile):
                    errors.append(
                        f"{rel_path}:{line_idx}: 'sub-millisecond' allowed only inside generated block with ef and percentile"
                    )

            # Special phrase: L2SqrSIMD16Ext
            if "l2sqrsimd16ext" in line_lower:
                is_hnswlib = "hnswlib" in line_lower
                has_perf_claim = any(
                    word in line_lower
                    for word in ["fast", "faster", "advantage", "overhead", "speed", "throughput", "latency", "boost", "outperform", "gain", "efficiency"]
                )
                if not (is_hnswlib and not has_perf_claim):
                    errors.append(
                        f"{rel_path}:{line_idx}: 'L2SqrSIMD16Ext' allowed only when stating it is hnswlib's with no performance claim"
                    )

            # Protected phrases: root cause & hypothesis confirmed
            has_rc = "root cause" in line_lower
            has_hc = "hypothesis confirmed" in line_lower
            if has_rc or has_hc:
                phrase_name = "root cause" if has_rc else "hypothesis confirmed"
                if rel_path != "docs/INVESTIGATIONS.md":
                    errors.append(
                        f"{rel_path}:{line_idx}: Protected phrase '{phrase_name}' allowed only in docs/INVESTIGATIONS.md"
                    )
                else:
                    # Look ahead for evidence: <path> on next non-empty line
                    found_evidence = False
                    for next_idx in range(line_idx, min(line_idx + 4, len(lines))):
                        next_line = lines[next_idx].strip()
                        if not next_line:
                            continue
                        ev_match = re.match(r'evidence:\s*(.+)', next_line, re.IGNORECASE)
                        if ev_match:
                            ev_path = ev_match.group(1).strip()
                            # Strip markdown backticks or brackets if any
                            ev_path = ev_path.strip('`[]()')
                            full_ev = os.path.join(root_dir, ev_path)
                            if os.path.exists(full_ev):
                                found_evidence = True
                            else:
                                errors.append(
                                    f"{rel_path}:{next_idx + 1}: Evidence path '{ev_path}' does not exist on disk"
                                )
                                found_evidence = True  # Avoid double reporting
                            break
                        else:
                            break
                    if not found_evidence:
                        errors.append(
                            f"{rel_path}:{line_idx}: '{phrase_name}' must be followed by 'evidence: <path>' line pointing to an existing file"
                        )

            # Numbers with units & recall decimals outside generated blocks
            if not in_generated and not in_corrections:
                # Find all numbers with units
                unit_matches = NUMBER_WITH_UNIT_RE.findall(line)
                decimal_matches = RECALL_DECIMAL_RE.findall(line)
                candidates = unit_matches + decimal_matches

                for val in candidates:
                    # Check if this occurrence matches any allowlist entry
                    allowed = False
                    for entry in allowlist_entries:
                        if entry["file"] == rel_path and entry["regex"].search(line):
                            if entry["value"] in line:
                                allowed = True
                                break
                    if not allowed:
                        errors.append(
                            f"{rel_path}:{line_idx}: Unverified number/metric outside generated block: '{val}' (not in scoped allowlist)"
                        )

    return errors


def main():
    print("Running Strict Claims and Consistency Checker (Round 5)...")
    errors = check_claims()

    if errors:
        print(f"\nFAILED: Found {len(errors)} issue(s):")
        for err in errors:
            print(f"  FAIL: {err}")
        return 1
    else:
        print("\nPASSED: All claims, scoped allowlists, and evidence paths verified successfully.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
