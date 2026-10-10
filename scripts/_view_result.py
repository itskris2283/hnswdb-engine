import json, sys, glob, os

for f in sorted(glob.glob("results/raw/bench_B_*.json")):
    d = json.load(open(f))
    ds = d["dataset"]
    print(f"\n=== {ds} ({os.path.basename(f)}) ===")
    for eng_name, eng_data in d["engines"].items():
        for sk, sd in eng_data.items():
            if not isinstance(sd, dict) or "ef_results" not in sd:
                continue
            print(f"  {eng_name} build={sd['build_time_s']}s")
            for e in sd["ef_results"]:
                print(f"    ef={e['ef_search']:>3}: R={e['recall_at_k']:.4f} AR={e['approx_ratio']:.4f} p50={e['latency']['p50']:.3f}ms")
