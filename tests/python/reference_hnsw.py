"""
Reference implementation of HNSW core algorithms in pure Python / NumPy.
Used for conformance testing against the C++ HNSWDB implementation.
Directly implements Algorithm 4 (SELECT-NEIGHBORS-HEURISTIC) and
Algorithm 2 (SEARCH-LAYER) from Malkov & Yashunin (2018).
"""

import heapq
import numpy as np
from typing import List, Tuple, Set


def reference_select_neighbors_heuristic(
    query_vec: np.ndarray,
    candidates: List[Tuple[float, int]],
    vectors: np.ndarray,
    max_links: int,
    keep_pruned: bool = False,
) -> List[int]:
    """
    Algorithm 4: Heuristic neighbor selection from Malkov & Yashunin (2018).
    candidates: list of (distance_to_query, index)
    vectors: full matrix of vectors [N, dim]
    max_links: maximum number of neighbors to select (M)
    keep_pruned: whether to keep discarded elements if fewer than max_links selected
    """
    if not candidates:
        return []

    # Sort candidates by distance to query ascending, then by index for determinism
    sorted_cands = sorted(candidates, key=lambda c: (c[0], c[1]))

    # If keep_pruned is False and candidate list <= max_links, return all candidates
    if not keep_pruned and len(sorted_cands) <= max_links:
        return [c[1] for c in sorted_cands]

    return_list: List[int] = []
    discarded_list: List[Tuple[float, int]] = []

    for dist_to_q, e_idx in sorted_cands:
        if len(return_list) >= max_links:
            break

        e_vec = vectors[e_idx]
        is_good = True

        for r_idx in return_list:
            r_vec = vectors[r_idx]
            # Squared L2 distance between candidate and already selected neighbor
            diff = e_vec - r_vec
            dist_to_r = float(np.sum(diff * diff))

            if dist_to_r < dist_to_q:
                is_good = False
                break

        if is_good:
            return_list.append(e_idx)
        elif keep_pruned:
            discarded_list.append((dist_to_q, e_idx))

    if keep_pruned and len(return_list) < max_links:
        for _, disc_idx in discarded_list:
            if len(return_list) >= max_links:
                break
            return_list.append(disc_idx)

    return return_list


def reference_search_layer(
    query_vec: np.ndarray,
    entry_point: int,
    ef: int,
    adj_list: List[List[int]],
    vectors: np.ndarray,
) -> List[Tuple[float, int]]:
    """
    Algorithm 2: SEARCH-LAYER from Malkov & Yashunin (2018).
    Traverses adj_list using greedy best-first search maintaining ef candidates.
    Returns sorted list of (distance, index) closest to query_vec.
    """
    ep_diff = query_vec - vectors[entry_point]
    ep_dist = float(np.sum(ep_diff * ep_diff))

    visited: Set[int] = {entry_point}

    # candidates: min-heap of (dist, idx)
    candidates: List[Tuple[float, int]] = [(ep_dist, entry_point)]

    # w_results: max-heap of (-dist, idx) to keep track of ef nearest
    w_results: List[Tuple[float, int]] = [(-ep_dist, entry_point)]

    while candidates:
        curr_dist, curr_idx = heapq.heappop(candidates)
        furthest_dist = -w_results[0][0]

        if curr_dist > furthest_dist:
            break

        for nbr in adj_list[curr_idx]:
            if nbr not in visited:
                visited.add(nbr)
                diff = query_vec - vectors[nbr]
                nbr_dist = float(np.sum(diff * diff))

                furthest_dist = -w_results[0][0]
                if len(w_results) < ef or nbr_dist < furthest_dist:
                    heapq.heappush(candidates, (nbr_dist, nbr))
                    heapq.heappush(w_results, (-nbr_dist, nbr))

                    if len(w_results) > ef:
                        heapq.heappop(w_results)

    # Sort results ascending by distance
    results = [(-d, idx) for d, idx in w_results]
    results.sort(key=lambda x: (x[0], x[1]))
    return results
