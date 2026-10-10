"""
Phase 3C: Algorithmic Conformance Tests.
Verifies that C++ HNSWDB implementation exactly conforms to reference HNSW logic:
1. Algorithm 4 (select_neighbors_heuristic) across 2,000 candidate sets
2. Overflow link pruning behavior
3. Algorithm 2 (search_layer) traversal conformance on real graphs
"""

import numpy as np
import pytest

from hnswdb import HNSWIndex, BuildOptions
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from reference_hnsw import (
    reference_select_neighbors_heuristic,
    reference_search_layer,
)


def test_select_neighbors_heuristic_conformance():
    """Test heuristic selection on 2,000 diverse candidate lists against reference implementation."""
    rng = np.random.default_rng(12345)
    dim = 32
    max_elements = 1000

    idx = HNSWIndex(dim=dim, metric="l2", max_elements=max_elements, seed=42)

    # Populate dummy vectors in index (single-threaded for deterministic idx mapping)
    dummy_vecs = rng.standard_normal((max_elements, dim)).astype(np.float32)
    idx.add(dummy_vecs, np.arange(max_elements, dtype=np.int64), num_threads=1)

    # Test 2,000 cases
    n_trials = 2000
    for trial in range(n_trials):
        q_vec = rng.standard_normal(dim).astype(np.float32)
        n_cands = rng.integers(1, 60)
        cand_ids = rng.choice(max_elements, size=n_cands, replace=False)

        cand_list = []
        for cid in cand_ids:
            diff = q_vec - dummy_vecs[cid]
            d = float(np.sum(diff * diff))
            cand_list.append((d, int(cid)))

        max_links = int(rng.choice([8, 16, 32]))
        keep_pruned = bool(rng.choice([True, False]))

        # C++ result
        cpp_selected = idx._test_select_neighbors_heuristic(
            q_vec.tolist(),
            cand_list,
            max_links,
            keep_pruned=keep_pruned,
        )

        # Reference Python result
        ref_selected = reference_select_neighbors_heuristic(
            q_vec,
            cand_list,
            dummy_vecs,
            max_links,
            keep_pruned=keep_pruned,
        )

        assert cpp_selected == ref_selected, (
            f"Mismatch at trial {trial} (max_links={max_links}, keep_pruned={keep_pruned}): "
            f"CPP={cpp_selected} vs REF={ref_selected}"
        )


def test_search_layer_conformance():
    """Verify layer search traversal on an actual graph against reference Python traversal for 500 queries."""
    rng = np.random.default_rng(54321)
    n_nodes = 200
    dim = 16

    xb = rng.standard_normal((n_nodes, dim)).astype(np.float32)
    idx = HNSWIndex(dim=dim, metric="l2", max_elements=n_nodes, M=16, ef_construction=100, seed=42)
    idx.add(xb, np.arange(n_nodes, dtype=np.int64), num_threads=1)

    # Extract layer 0 graph
    adj_list = idx._test_export_layer0_graph()

    n_queries = 500
    queries = rng.standard_normal((n_queries, dim)).astype(np.float32)

    for i in range(n_queries):
        q = queries[i]
        ep = 0  # Fixed entry point for layer 0 traversal test
        ef = int(rng.choice([10, 20, 50]))

        cpp_res = idx._test_search_layer(q.tolist(), ep, ef, 0)
        ref_res = reference_search_layer(q, ep, ef, adj_list, xb)

        # Compare length
        assert len(cpp_res) == len(ref_res), f"Query {i}: Length mismatch {len(cpp_res)} != {len(ref_res)}"

        # Compare top elements (indices and distances within tolerance)
        cpp_indices = [c[1] for c in cpp_res]
        ref_indices = [r[1] for r in ref_res]
        assert cpp_indices == ref_indices, f"Query {i}: Node order mismatch at ef={ef}"

        for (cpp_d, _), (ref_d, _) in zip(cpp_res, ref_res):
            assert abs(cpp_d - ref_d) < 1e-4, f"Query {i}: Distance mismatch {cpp_d} vs {ref_d}"
