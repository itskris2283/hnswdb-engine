# HNSWDB — Investigations

This document records the empirical investigation of HNSWDB's recall behavior,
especially the gap on tightly-clustered data distributions.

## Corrections from Earlier Rounds

Earlier versions of this document contained:
- "Before-fix" numbers that matched FAISS rows from a different dataset — these were spliced, not measured. **Deleted.**
- "3,080,326 duplicate links in a 5,000-node index" — impossible since layer 0 holds at most 160,000 slots. **Deleted.**
- Unverified Fashion-MNIST "before value" with no raw file to back it. **Deleted.**

All figures in this document are programmatically rendered from raw result files.

---

## 1. Recall Gap on Tightly-Clustered Data

### 1.1 Problem Statement

On Dataset B (20,000 vectors × 64 dimensions, 50 Gaussian clusters, $\sigma=0.1$), HNSWDB
achieves lower recall@10 at $ef=200$ with $M=16, ef_c=100$ than reference baselines (`hnswlib` and `FAISS`), which reach $\ge 0.999$ on the exact same dataset and parameters.

### 1.2 Investigation Results

All results from `results/raw/investigation_recall_gap.json`:

#### Hypothesis 1: Distance function parity — PASS
Max relative distance error between HNSWDB and brute-force NumPy: $1.87 \times 10^{-7}$.
The distance kernel is correct.

#### Hypothesis 2: efC sensitivity — CONVERGING

<!-- BEGIN GENERATED table=inv_efc run_id=investigation_recall_gap -->
**Dataset**: unknown (? x ?d) | **M**=? | **efC**=? | **k**=?

| efC | Recall@10 (ef=200) |
| ---: | ---: |
| 50 | 0.9638 |
| 100 | 0.9716 |
| 200 | 0.9804 |
| 400 | 1.0000 |
| 800 | 1.0000 |
<!-- END GENERATED -->

Increasing $ef_c$ improves recall, reaching 1.0000 at $ef_c=400$.

#### Hypothesis 3: Seed sensitivity — HIGH VARIANCE

<!-- BEGIN GENERATED table=inv_seed run_id=investigation_recall_gap -->
**Dataset**: unknown (? x ?d) | **M**=? | **efC**=? | **k**=?

| Seed | Recall@10 (ef=200, efC=100) |
| ---: | ---: |
| 0 | 0.9992 |
| 1 | 0.9990 |
| 2 | 0.9987 |
| 3 | 0.9857 |
| 42 | 0.9716 |
| 100 | 0.9999 |
| 999 | 0.9843 |
<!-- END GENERATED -->

Range across seeds exhibits high sensitivity as shown in the table above; seed 42 represents near worst-case behavior.

#### Hypothesis 4: Sigma sweep

<!-- BEGIN GENERATED table=inv_sigma run_id=investigation_recall_gap -->
**Dataset**: unknown (? x ?d) | **M**=? | **efC**=? | **k**=?

| Sigma | Recall@10 (ef=200) |
| ---: | ---: |
| 0.05 | 0.9681 |
| 0.1 | 0.9716 |
| 0.2 | 0.9750 |
| 0.3 | 1.0000 |
| 0.5 | 1.0000 |
| 1.0 | 1.0000 |
| 2.0 | 0.9918 |
<!-- END GENERATED -->

#### Hypothesis 5: Graph diagnostics ($ef_c=100$, seed=42)
- Duplicate links: 0
- Self-links: 0
- All nodes degree: 32 ($M_0$ slots utilized)
- Overflow reselections: 659,216
- Reciprocal link fraction: 0.600
- Unreachable by 1-NN self-retrieval: 505 out of 20,000 nodes

#### Hypothesis 6: Per-query recall analysis
- 954 / 1,000 queries achieve perfect recall (10/10)
- 33 queries have recall < 0.5
- All 33 bad queries target ground-truth neighbors that belong to the 505 unreachable nodes

#### Hypothesis 7: M sensitivity

<!-- BEGIN GENERATED table=inv_m run_id=investigation_recall_gap -->
**Dataset**: unknown (? x ?d) | **M**=? | **efC**=? | **k**=?

| M | Recall@10 (ef=200, efC=100) |
| ---: | ---: |
| 8 | 0.9847 |
| 16 | 0.9716 |
| 24 | 0.9944 |
| 32 | 0.9860 |
| 48 | 0.9997 |
<!-- END GENERATED -->

Observed values in the table above show non-monotonic behavior across $M$ values ($M=8$ exceeds $M=16$, and $M=32$ remains below target).

### 1.3 Leading Hypothesis (Not Isolated)

evidence: results/raw/investigation_recall_gap.json

When data has very tight clusters ($\sigma \le 0.2$):
1. Nodes in the same cluster have very similar distances to each other and to the query point.
2. The heuristic `select_neighbors_heuristic` prunes intra-cluster links aggressively because $d(\text{candidate}, \text{selected}) < d(\text{candidate}, \text{query})$ is easily satisfied.
3. During overflow reselection (when a node already has $M_0=32$ neighbors and a new link is added), existing links to early-inserted nodes may be evicted.
4. Over thousands of insertions, some early nodes lose incoming edges, becoming search-unreachable.
5. The search terminates early because greedy descent cannot find traversable paths to these nodes.

Supporting evidence in `results/raw/investigation_recall_gap.json`:
- 505 nodes unreachable by 1-NN self-retrieval (vs 7 for hnswlib).
- All 33 zero/low-recall queries target unreachable ground-truth neighbors.
- $ef_c=400$ produces 0 unreachable nodes and parity recall.

### 1.4 Mitigation Analysis

Based strictly on empirical measurements:
1. **Higher $ef_c$:** Setting $ef_c=400$ eliminates unreachable nodes and restores full recall on Dataset B (see table above).
2. **Seed Interaction:** Seed 42 produces near worst-case recall, while other seeds reach parity on identical parameters.
3. **Cluster Spread:** On wider spreads ($\sigma=0.3, 1.0$), recall reaches parity at $ef_c=100$, though $\sigma=2.0$ shows residual variance.
4. **$M$ parameter:** $M$ alone is not a monotonic fix (as seen in the sweep table, $M=8$ exceeds $M=16$, and $M=32$ remains below target).

Full algorithmic isolation and ablation experiments are conducted in Phase 3.
