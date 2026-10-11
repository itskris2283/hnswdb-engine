#include "hnswdb/hnsw_index.h"
#include "hnswdb/serialization.h"
#include <algorithm>
#include <cmath>
#include <queue>
#include <unordered_set>
#include <chrono>
#include <cstring>
#include <future>
#include <thread>
#include <iostream>

namespace hnswdb {

HNSWIndex::HNSWIndex(std::size_t dim,
                     Metric metric,
                     std::size_t max_elements,
                     std::size_t M,
                     std::size_t ef_construction,
                     std::size_t ef_search,
                     uint64_t seed,
                     BuildOptions build_options)
    : dim_(dim),
      metric_(metric),
      max_elements_(max_elements),
      M_(M),
      M0_(2 * M),
      Mmax_(M),
      ef_construction_(ef_construction),
      ef_search_(ef_search),
      seed_(seed),
      rng_(seed),
      build_options_(std::move(build_options)) {
    if (dim_ < 1 || dim_ > 4096) {
        throw InvalidArgument("Dimension must be between 1 and 4096, got " + std::to_string(dim_));
    }
    if (M_ < 2 || M_ > 128) {
        throw InvalidArgument("M must be between 2 and 128, got " + std::to_string(M_));
    }
    if (max_elements_ == 0) {
        throw InvalidArgument("max_elements must be greater than 0");
    }

    mL_ = 1.0 / std::log(static_cast<double>(M_));

    if (metric_ == Metric::L2) {
        dist_func_ = distance::get_l2sq_func();
    } else {
        dist_func_ = distance::get_dot_func();
    }

    vectors_.resize(max_elements_ * dim_);
    labels_.resize(max_elements_);
    levels_.resize(max_elements_, 0);
    deleted_.resize(max_elements_, 0);

    layer0_links_.resize(max_elements_ * (1 + M0_), 0);
    upper_links_.resize(max_elements_);
    in_degrees_l0_.resize(max_elements_, 0);

    striped_locks_ = std::make_unique<SpinLock[]>(NUM_STRIPES);

    std::size_t num_pool = std::max(4u, std::thread::hardware_concurrency() * 2);
    visited_pool_ = std::make_unique<VisitedListPool>(num_pool, max_elements_);
}

HNSWIndex::HNSWIndex(HNSWIndex&& other) noexcept {
    *this = std::move(other);
}

HNSWIndex& HNSWIndex::operator=(HNSWIndex&& other) noexcept {
    if (this != &other) {
        std::unique_lock<std::shared_mutex> lock(structure_lock_);
        std::unique_lock<std::shared_mutex> other_lock(other.structure_lock_);

        dim_ = other.dim_;
        metric_ = other.metric_;
        max_elements_ = other.max_elements_;
        M_ = other.M_;
        M0_ = other.M0_;
        Mmax_ = other.Mmax_;
        ef_construction_ = other.ef_construction_;
        ef_search_.store(other.ef_search_.load());
        mL_ = other.mL_;
        seed_ = other.seed_;
        dist_func_ = other.dist_func_;
        build_options_ = std::move(other.build_options_);

        vectors_ = std::move(other.vectors_);
        labels_ = std::move(other.labels_);
        levels_ = std::move(other.levels_);
        deleted_ = std::move(other.deleted_);
        layer0_links_ = std::move(other.layer0_links_);
        upper_links_ = std::move(other.upper_links_);
        in_degrees_l0_ = std::move(other.in_degrees_l0_);
        label_to_idx_ = std::move(other.label_to_idx_);
        cur_element_count_.store(other.cur_element_count_.load());
        live_count_.store(other.live_count_.load());
        entry_point_.store(other.entry_point_.load());
        max_level_.store(other.max_level_.load());

        metadata_store_ = std::move(other.metadata_store_);

        striped_locks_ = std::move(other.striped_locks_);
        visited_pool_ = std::move(other.visited_pool_);
    }
    return *this;
}

int HNSWIndex::get_random_level() {
    std::lock_guard<std::mutex> lock(rng_mutex_);
    double r = uniform_dist_(rng_);
    if (r <= 0.0) r = 1e-12;
    int lvl = static_cast<int>(std::floor(-std::log(r) * mL_));
    return lvl;
}

float HNSWIndex::compute_dist(const float* a, const float* b) const {
    if (metric_ == Metric::L2) {
        return dist_func_(a, b, dim_);
    } else if (metric_ == Metric::IP) {
        return -dist_func_(a, b, dim_);
    } else { // Cosine
        return 1.0f - dist_func_(a, b, dim_);
    }
}

float HNSWIndex::compute_dist_by_idx(const float* query, idx_t idx) const {
    const float* vec = vectors_.data() + idx * dim_;
    return compute_dist(query, vec);
}

uint32_t* HNSWIndex::get_layer0_link_count_ptr(idx_t idx) {
    return &layer0_links_[idx * (1 + M0_)];
}

const uint32_t* HNSWIndex::get_layer0_link_count_ptr(idx_t idx) const {
    return &layer0_links_[idx * (1 + M0_)];
}

uint32_t* HNSWIndex::get_layer0_links_ptr(idx_t idx) {
    return &layer0_links_[idx * (1 + M0_) + 1];
}

const uint32_t* HNSWIndex::get_layer0_links_ptr(idx_t idx) const {
    return &layer0_links_[idx * (1 + M0_) + 1];
}

void HNSWIndex::get_node_links(idx_t idx, int layer, std::vector<idx_t>& out_links) const {
    SpinLockGuard guard(get_stripe_lock(idx));
    if (layer == 0) {
        uint32_t count = *get_layer0_link_count_ptr(idx);
        const uint32_t* links = get_layer0_links_ptr(idx);
        out_links.assign(links, links + count);
    } else {
        if (layer - 1 < static_cast<int>(upper_links_[idx].size())) {
            out_links = upper_links_[idx][layer - 1];
        } else {
            out_links.clear();
        }
    }
}

void HNSWIndex::set_node_links(idx_t idx, int layer, const std::vector<idx_t>& links) {
    SpinLockGuard guard(get_stripe_lock(idx));
    if (layer == 0) {
        uint32_t count = static_cast<uint32_t>(std::min(links.size(), M0_));
        *get_layer0_link_count_ptr(idx) = count;
        uint32_t* dst = get_layer0_links_ptr(idx);
        std::memcpy(dst, links.data(), count * sizeof(uint32_t));
    } else {
        if (layer - 1 >= static_cast<int>(upper_links_[idx].size())) {
            upper_links_[idx].resize(layer);
        }
        uint32_t count = static_cast<uint32_t>(std::min(links.size(), Mmax_));
        upper_links_[idx][layer - 1].assign(links.begin(), links.begin() + count);
    }
}

std::vector<HNSWIndex::Candidate> HNSWIndex::search_layer(
    const float* query_vec,
    idx_t entry_point,
    std::size_t ef,
    int layer,
    VisitedList* visited,
    bool is_build) const {

    std::priority_queue<Candidate, std::vector<Candidate>, std::greater<Candidate>> candidates;
    std::priority_queue<Candidate, std::vector<Candidate>, std::less<Candidate>> w_results;

    if (is_build) {
        distance_computations_build_++;
    } else {
        distance_computations_query_++;
    }
    float init_dist = compute_dist_by_idx(query_vec, entry_point);
    visited->mark_visited(entry_point);

    candidates.push({init_dist, entry_point});
    w_results.push({init_dist, entry_point});

    uint32_t local_nbrs[512];

    while (!candidates.empty()) {
        Candidate curr = candidates.top();
        candidates.pop();

        Candidate furthest_result = w_results.top();
        if (curr.dist > furthest_result.dist && w_results.size() >= ef) {
            break;
        }

        uint32_t nbr_count = 0;
        if (layer == 0) {
            SpinLockGuard guard(get_stripe_lock(curr.idx));
            nbr_count = *get_layer0_link_count_ptr(curr.idx);
            if (nbr_count > 512) nbr_count = 512;
            const uint32_t* links = get_layer0_links_ptr(curr.idx);
            std::memcpy(local_nbrs, links, nbr_count * sizeof(uint32_t));
        } else {
            SpinLockGuard guard(get_stripe_lock(curr.idx));
            if (layer - 1 < static_cast<int>(upper_links_[curr.idx].size())) {
                const auto& l_links = upper_links_[curr.idx][layer - 1];
                nbr_count = static_cast<uint32_t>(std::min<size_t>(l_links.size(), 512));
                std::memcpy(local_nbrs, l_links.data(), nbr_count * sizeof(uint32_t));
            }
        }

        for (uint32_t i = 0; i < nbr_count; ++i) {
            idx_t nbr = local_nbrs[i];
            if (visited->mark_visited(nbr)) {
                distance::prefetch_vector(vectors_.data() + nbr * dim_, dim_);
                if (is_build) {
                    distance_computations_build_++;
                } else {
                    distance_computations_query_++;
                }
                float nbr_dist = compute_dist_by_idx(query_vec, nbr);

                if (w_results.size() < ef || nbr_dist < w_results.top().dist) {
                    candidates.push({nbr_dist, nbr});
                    w_results.push({nbr_dist, nbr});

                    if (w_results.size() > ef) {
                        w_results.pop();
                    }
                }
            }
        }
    }

    std::vector<Candidate> result_vec;
    result_vec.reserve(w_results.size());
    while (!w_results.empty()) {
        result_vec.push_back(w_results.top());
        w_results.pop();
    }
    std::reverse(result_vec.begin(), result_vec.end());
    return result_vec;
}

std::vector<HNSWIndex::Candidate> HNSWIndex::search_layer_multi(
    const float* query_vec,
    const std::vector<idx_t>& entry_points,
    std::size_t ef,
    int layer,
    VisitedList* visited,
    bool is_build) const {

    std::priority_queue<Candidate, std::vector<Candidate>, std::greater<Candidate>> candidates;
    std::priority_queue<Candidate, std::vector<Candidate>, std::less<Candidate>> w_results;

    for (idx_t ep : entry_points) {
        if (visited->mark_visited(ep)) {
            if (is_build) {
                distance_computations_build_++;
            } else {
                distance_computations_query_++;
            }
            float dist = compute_dist_by_idx(query_vec, ep);
            candidates.push({dist, ep});
            w_results.push({dist, ep});
            if (w_results.size() > ef) {
                w_results.pop();
            }
        }
    }

    uint32_t local_nbrs[512];

    while (!candidates.empty()) {
        Candidate curr = candidates.top();
        candidates.pop();

        Candidate furthest_result = w_results.top();
        if (curr.dist > furthest_result.dist && w_results.size() >= ef) {
            break;
        }

        uint32_t nbr_count = 0;
        if (layer == 0) {
            SpinLockGuard guard(get_stripe_lock(curr.idx));
            nbr_count = *get_layer0_link_count_ptr(curr.idx);
            if (nbr_count > 512) nbr_count = 512;
            const uint32_t* links = get_layer0_links_ptr(curr.idx);
            std::memcpy(local_nbrs, links, nbr_count * sizeof(uint32_t));
        } else {
            SpinLockGuard guard(get_stripe_lock(curr.idx));
            if (layer - 1 < static_cast<int>(upper_links_[curr.idx].size())) {
                const auto& l_links = upper_links_[curr.idx][layer - 1];
                nbr_count = static_cast<uint32_t>(std::min<size_t>(l_links.size(), 512));
                std::memcpy(local_nbrs, l_links.data(), nbr_count * sizeof(uint32_t));
            }
        }

        for (uint32_t i = 0; i < nbr_count; ++i) {
            idx_t nbr = local_nbrs[i];
            if (visited->mark_visited(nbr)) {
                distance::prefetch_vector(vectors_.data() + nbr * dim_, dim_);
                if (is_build) {
                    distance_computations_build_++;
                } else {
                    distance_computations_query_++;
                }
                float nbr_dist = compute_dist_by_idx(query_vec, nbr);

                if (w_results.size() < ef || nbr_dist < w_results.top().dist) {
                    candidates.push({nbr_dist, nbr});
                    w_results.push({nbr_dist, nbr});

                    if (w_results.size() > ef) {
                        w_results.pop();
                    }
                }
            }
        }
    }

    std::vector<Candidate> result_vec;
    result_vec.reserve(w_results.size());
    while (!w_results.empty()) {
        result_vec.push_back(w_results.top());
        w_results.pop();
    }
    std::reverse(result_vec.begin(), result_vec.end());
    return result_vec;
}


std::vector<idx_t> HNSWIndex::select_neighbors_heuristic(
    const float* query_vec,
    std::vector<Candidate>& candidates,
    std::size_t max_links,
    bool keep_pruned) const {

    heuristic_selection_calls_++;

    if (candidates.empty()) {
        return {};
    }

    // Sort candidates by ascending distance
    std::sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) {
        if (a.dist != b.dist) return a.dist < b.dist;
        return a.idx < b.idx;
    });

#ifndef HNSWDB_LEGACY_DUPLICATE_BUG
    // Remove duplicates in-place
    size_t unique_count = 0;
    for (size_t i = 0; i < candidates.size(); ++i) {
        if (i == 0 || candidates[i].idx != candidates[unique_count - 1].idx) {
            candidates[unique_count++] = candidates[i];
        }
    }
    candidates.resize(unique_count);
#endif

    if (candidates.size() <= max_links && !keep_pruned) {
        std::vector<idx_t> result;
        result.reserve(candidates.size());
        for (const auto& c : candidates) {
            result.push_back(c.idx);
        }
        return result;
    }

    std::vector<idx_t> return_list;
    return_list.reserve(max_links);
    std::vector<Candidate> discarded_list;
    if (keep_pruned) {
        discarded_list.reserve(candidates.size());
    }

    for (const auto& current : candidates) {
        if (return_list.size() >= max_links) break;

        bool is_good = true;
        const float* current_vec = vectors_.data() + current.idx * dim_;

        for (idx_t selected_idx : return_list) {
            const float* selected_vec = vectors_.data() + selected_idx * dim_;
            distance_computations_build_++;
            float dist_to_selected = compute_dist(current_vec, selected_vec);

            if (dist_to_selected < current.dist) {
                is_good = false;
                break;
            }
        }


        if (is_good) {
            return_list.push_back(current.idx);
        } else if (keep_pruned) {
            discarded_list.push_back(current);
        }
    }

    if (keep_pruned && return_list.size() < max_links) {
        for (const auto& discarded : discarded_list) {
            if (return_list.size() >= max_links) break;
#ifndef HNSWDB_LEGACY_DUPLICATE_BUG
            bool already_present = false;
            for (idx_t sel : return_list) {
                if (sel == discarded.idx) {
                    already_present = true;
                    break;
                }
            }
            if (!already_present) {
                return_list.push_back(discarded.idx);
            }
#else
            return_list.push_back(discarded.idx);
#endif
        }
    }

    return return_list;
}

void HNSWIndex::add(const float* vector, label_t label, const MetaMap* metadata) {
    if (label < 0) {
        throw InvalidArgument("Label must be non-negative, got " + std::to_string(label));
    }
    if (!distance::validate_vector(vector, dim_)) {
        throw InvalidVector("Vector contains NaN or Inf");
    }

    std::vector<float> norm_buf;
    const float* vec_to_store = vector;
    if (metric_ == Metric::Cosine) {
        norm_buf.resize(dim_);
        distance::normalize_vector(vector, norm_buf.data(), dim_);
        vec_to_store = norm_buf.data();
    }

    std::shared_lock<std::shared_mutex> struct_lock(structure_lock_);

    idx_t new_idx = static_cast<idx_t>(cur_element_count_.fetch_add(1));
    if (new_idx >= max_elements_) {
        cur_element_count_.fetch_sub(1);
        throw CapacityExceeded("Index capacity exceeded: max_elements = " + std::to_string(max_elements_));
    }

    std::memcpy(vectors_.data() + new_idx * dim_, vec_to_store, dim_ * sizeof(float));
    labels_[new_idx] = label;
    deleted_[new_idx] = 0;

    {
        std::lock_guard<std::mutex> map_lock(label_map_mutex_);
        auto it = label_to_idx_.find(label);
        if (it != label_to_idx_.end()) {
            idx_t existing_idx = it->second;
            if (!deleted_[existing_idx]) {
                deleted_[existing_idx] = 1;
                --live_count_;
            }
        }
        label_to_idx_[label] = new_idx;
        ++live_count_;
    }

    if (metadata) {
        metadata_store_.add_metadata(label, *metadata);
    }

    int node_level = get_random_level();
    levels_[new_idx] = static_cast<uint8_t>(node_level);

    if (node_level > 0) {
        upper_links_[new_idx].resize(node_level);
    }

    if (new_idx == 0) {
        std::lock_guard<std::mutex> ep_lock(entry_point_mutex_);
        entry_point_ = 0;
        max_level_ = node_level;
        return;
    }

    idx_t curr_obj;
    int curr_max_level;
    {
        std::lock_guard<std::mutex> ep_lock(entry_point_mutex_);
        curr_obj = entry_point_.load();
        curr_max_level = max_level_.load();
    }

    // Greedy descent from curr_max_level down to node_level + 1
    distance_computations_build_++;
    float cur_dist = compute_dist_by_idx(vec_to_store, curr_obj);
    for (int l = curr_max_level; l > node_level; --l) {
        bool changed = true;
        std::vector<idx_t> links;
        while (changed) {
            changed = false;
            get_node_links(curr_obj, l, links);
            for (idx_t nbr : links) {
                distance_computations_build_++;
                float d = compute_dist_by_idx(vec_to_store, nbr);
                if (d < cur_dist) {
                    cur_dist = d;
                    curr_obj = nbr;
                    changed = true;
                }
            }
        }
    }

    // Connect layers down to 0
    std::vector<idx_t> enter_points;
    enter_points.push_back(curr_obj);

    for (int l = std::min(node_level, curr_max_level); l >= 0; --l) {
        VisitedListGuard visited_guard(*visited_pool_);
        std::vector<Candidate> candidates;
        if (build_options_.layer_entry == "all" && enter_points.size() > 1) {
            candidates = search_layer_multi(vec_to_store, enter_points, ef_construction_, l, visited_guard.get(), true);
        } else {
            candidates = search_layer(vec_to_store, enter_points.empty() ? curr_obj : enter_points[0], ef_construction_, l, visited_guard.get(), true);
        }

        std::size_t M_l = (l == 0) ? ((build_options_.new_node_degree == "M") ? M_ : M0_) : Mmax_;

        std::vector<idx_t> selected_neighbors = select_neighbors_heuristic(vec_to_store, candidates, M_l, build_options_.keep_pruned);

        set_node_links(new_idx, l, selected_neighbors);
        if (l == 0) {
            for (idx_t nbr : selected_neighbors) {
                if (nbr < in_degrees_l0_.size()) {
                    in_degrees_l0_[nbr]++;
                }
            }
        }

        for (idx_t nbr : selected_neighbors) {
            SpinLockGuard guard(get_stripe_lock(nbr));

            if (l == 0) {
                uint32_t count = *get_layer0_link_count_ptr(nbr);
                uint32_t* links = get_layer0_links_ptr(nbr);

                bool already_linked = false;
#ifndef HNSWDB_LEGACY_DUPLICATE_BUG
                for (uint32_t i = 0; i < count; ++i) {
                    if (links[i] == new_idx) {
                        already_linked = true;
                        break;
                    }
                }
                if (already_linked) {
                    duplicate_insert_attempts_++;
                }
#endif

                if (!already_linked) {
                    if (count < M0_) {
                        links[count++] = new_idx;
                        *get_layer0_link_count_ptr(nbr) = count;
                        if (new_idx < in_degrees_l0_.size()) {
                            in_degrees_l0_[new_idx]++;
                        }
                    } else {
                        overflow_reselection_calls_++;
                        std::vector<Candidate> nbr_candidates;
                        nbr_candidates.reserve(count + 1);
                        const float* nbr_vec = vectors_.data() + nbr * dim_;
                        for (uint32_t i = 0; i < count; ++i) {
                            nbr_candidates.push_back({compute_dist_by_idx(nbr_vec, links[i]), links[i]});
                        }
                        nbr_candidates.push_back({compute_dist_by_idx(nbr_vec, new_idx), new_idx});

                        std::vector<idx_t> shrunk;
                        if (build_options_.overflow_selection == "closest") {
                            std::sort(nbr_candidates.begin(), nbr_candidates.end());
                            for (size_t si = 0; si < std::min<size_t>(M0_, nbr_candidates.size()); ++si) {
                                shrunk.push_back(nbr_candidates[si].idx);
                            }
                        } else {
                            shrunk = select_neighbors_heuristic(nbr_vec, nbr_candidates, M0_, build_options_.keep_pruned);
                        }

                        if (build_options_.protect_last_in_edge) {
                            std::unordered_set<idx_t> s_set(shrunk.begin(), shrunk.end());
                            std::vector<idx_t> r_set;
                            for (const auto& c : nbr_candidates) {
                                if (s_set.find(c.idx) == s_set.end()) {
                                    r_set.push_back(c.idx);
                                }
                            }

                            for (idx_t x : r_set) {
                                if (x < in_degrees_l0_.size() && in_degrees_l0_[x] == 1) {
                                    int best_s_idx = -1;
                                    float max_d = -1.0f;
                                    for (size_t si = 0; si < shrunk.size(); ++si) {
                                        idx_t y = shrunk[si];
                                        if (y < in_degrees_l0_.size() && in_degrees_l0_[y] >= 2) {
                                            float d = compute_dist_by_idx(nbr_vec, y);
                                            if (d > max_d) {
                                                max_d = d;
                                                best_s_idx = static_cast<int>(si);
                                            }
                                        }
                                    }
                                    if (best_s_idx >= 0) {
                                        idx_t evicted_y = shrunk[best_s_idx];
                                        shrunk[best_s_idx] = x;
                                        s_set.erase(evicted_y);
                                        s_set.insert(x);
                                    }
                                }
                            }
                        }

                        // Update in-degrees for changes between old links and new shrunk
                        std::unordered_set<idx_t> old_links(links, links + count);
                        std::unordered_set<idx_t> new_links(shrunk.begin(), shrunk.end());
                        for (idx_t old_l : old_links) {
                            if (new_links.find(old_l) == new_links.end() && old_l < in_degrees_l0_.size()) {
                                if (in_degrees_l0_[old_l] > 0) in_degrees_l0_[old_l]--;
                            }
                        }
                        for (idx_t new_l : new_links) {
                            if (old_links.find(new_l) == old_links.end() && new_l < in_degrees_l0_.size()) {
                                in_degrees_l0_[new_l]++;
                            }
                        }

                        *get_layer0_link_count_ptr(nbr) = static_cast<uint32_t>(shrunk.size());
                        std::memcpy(links, shrunk.data(), shrunk.size() * sizeof(uint32_t));
                    }
                }
            } else {
                if (l - 1 >= static_cast<int>(upper_links_[nbr].size())) {
                    upper_links_[nbr].resize(l);
                }
                auto& nbr_links = upper_links_[nbr][l - 1];

                bool already_linked = false;
#ifndef HNSWDB_LEGACY_DUPLICATE_BUG
                for (idx_t link_idx : nbr_links) {
                    if (link_idx == new_idx) {
                        already_linked = true;
                        break;
                    }
                }
                if (already_linked) {
                    duplicate_insert_attempts_++;
                }
#endif

                if (!already_linked) {
                    if (nbr_links.size() < Mmax_) {
                        nbr_links.push_back(new_idx);
                    } else {
                        overflow_reselection_calls_++;
                        std::vector<Candidate> nbr_candidates;
                        nbr_candidates.reserve(nbr_links.size() + 1);
                        const float* nbr_vec = vectors_.data() + nbr * dim_;
                        for (idx_t link_idx : nbr_links) {
                            nbr_candidates.push_back({compute_dist_by_idx(nbr_vec, link_idx), link_idx});
                        }
                        nbr_candidates.push_back({compute_dist_by_idx(nbr_vec, new_idx), new_idx});

                        nbr_links = select_neighbors_heuristic(nbr_vec, nbr_candidates, Mmax_, build_options_.keep_pruned);
                    }
                }
            }
        }

        if (!candidates.empty()) {
            enter_points.clear();
            if (build_options_.layer_entry == "all") {
                for (const auto& c : candidates) {
                    enter_points.push_back(c.idx);
                }
            } else {
                enter_points.push_back(candidates[0].idx);
            }
            curr_obj = candidates[0].idx;
        }
    }

    if (node_level > curr_max_level) {
        std::lock_guard<std::mutex> ep_lock(entry_point_mutex_);
        if (node_level > max_level_.load()) {
            max_level_ = node_level;
            entry_point_ = new_idx;
        }
    }
}

void HNSWIndex::add_batch(const float* vectors,
                          const label_t* labels,
                          std::size_t n,
                          const std::vector<MetaMap>* metadata,
                          std::size_t num_threads) {
    if (n == 0) return;
    if (num_threads == 0) {
        num_threads = std::max(1u, std::thread::hardware_concurrency());
    }

    if (num_threads == 1 || n < 100) {
        for (std::size_t i = 0; i < n; ++i) {
            const MetaMap* meta = (metadata && i < metadata->size()) ? &(*metadata)[i] : nullptr;
            add(vectors + i * dim_, labels[i], meta);
        }
        return;
    }

    std::vector<std::future<void>> futures;
    std::atomic<std::size_t> current_idx{0};

    for (std::size_t t = 0; t < num_threads; ++t) {
        futures.push_back(std::async(std::launch::async, [this, vectors, labels, metadata, n, &current_idx]() {
            while (true) {
                std::size_t idx = current_idx.fetch_add(1);
                if (idx >= n) break;
                const MetaMap* meta = (metadata && idx < metadata->size()) ? &(*metadata)[idx] : nullptr;
                add(vectors + idx * dim_, labels[idx], meta);
            }
        }));
    }

    for (auto& f : futures) {
        f.get();
    }
}

void HNSWIndex::set_metadata(label_t label, const MetaMap& metadata) {
    metadata_store_.add_metadata(label, metadata);
}

MetaMap HNSWIndex::get_metadata(label_t label) const {
    return metadata_store_.get_metadata(label);
}

bool HNSWIndex::has_metadata(label_t label) const {
    return metadata_store_.has_metadata(label);
}

double HNSWIndex::estimate_selectivity(const FilterExpr& filter, std::vector<label_t>& out_matching) const {
    std::size_t live = live_count_.load();
    if (live == 0) return 0.0;

    if (metadata_store_.can_use_inverted_index(filter)) {
        auto indexed_labels = metadata_store_.get_matching_labels_from_index(filter);
        out_matching.clear();
        out_matching.reserve(indexed_labels.size());
        {
            std::lock_guard<std::mutex> map_lock(label_map_mutex_);
            for (label_t l : indexed_labels) {
                auto it = label_to_idx_.find(l);
                if (it != label_to_idx_.end() && !deleted_[it->second]) {
                    out_matching.push_back(l);
                }
            }
        }
        return static_cast<double>(out_matching.size()) / static_cast<double>(live);
    }

    // Sample up to 1000 random live nodes
    std::size_t sample_size = std::min(live, std::size_t(1000));
    std::size_t matches = 0;
    std::size_t checked = 0;

    std::size_t total = cur_element_count_.load();
    for (std::size_t i = 0; i < total && checked < sample_size; ++i) {
        if (!deleted_[i]) {
            if (metadata_store_.evaluate(labels_[i], filter)) {
                ++matches;
            }
            ++checked;
        }
    }

    return checked > 0 ? (static_cast<double>(matches) / checked) : 0.0;
}

SearchResult HNSWIndex::brute_force_search_subset(const float* query, std::size_t k, const std::vector<label_t>& subset) const {
    struct DistPair {
        float dist;
        label_t label;
        bool operator<(const DistPair& other) const {
            return dist < other.dist; // Max-heap
        }
    };

    std::priority_queue<DistPair> max_heap;

    {
        std::lock_guard<std::mutex> map_lock(label_map_mutex_);
        for (label_t l : subset) {
            auto it = label_to_idx_.find(l);
            if (it != label_to_idx_.end()) {
                idx_t idx = it->second;
                if (!deleted_[idx]) {
                    float d = compute_dist_by_idx(query, idx);
                    if (max_heap.size() < k) {
                        max_heap.push({d, l});
                    } else if (d < max_heap.top().dist) {
                        max_heap.pop();
                        max_heap.push({d, l});
                    }
                }
            }
        }
    }

    SearchResult res;
    res.ids.resize(k, -1);
    res.dists.resize(k, std::numeric_limits<float>::infinity());

    std::size_t count = max_heap.size();
    for (std::size_t i = count; i > 0; --i) {
        auto top = max_heap.top();
        max_heap.pop();
        res.ids[i - 1] = top.label;
        res.dists[i - 1] = top.dist;
    }

    return res;
}

SearchResult HNSWIndex::search(const float* query,
                               std::size_t k,
                               std::size_t ef_search_override,
                               const FilterExpr* filter) const {
    if (k == 0) {
        throw InvalidArgument("k must be greater than 0");
    }
    if (!distance::validate_vector(query, dim_)) {
        throw InvalidVector("Query vector contains NaN or Inf");
    }

    std::vector<float> norm_buf;
    const float* q = query;
    if (metric_ == Metric::Cosine) {
        norm_buf.resize(dim_);
        distance::normalize_vector(query, norm_buf.data(), dim_);
        q = norm_buf.data();
    }

    std::shared_lock<std::shared_mutex> struct_lock(structure_lock_);

    std::size_t total_elements = cur_element_count_.load();
    if (total_elements == 0 || live_count_.load() == 0) {
        SearchResult empty_res;
        empty_res.ids.resize(k, -1);
        empty_res.dists.resize(k, std::numeric_limits<float>::infinity());
        return empty_res;
    }

    // Filtered search handling
    if (filter) {
        std::vector<label_t> matching_labels;
        double selectivity = estimate_selectivity(*filter, matching_labels);

        // If highly selective (<= 2%) or matching set <= 5000: exact brute-force over matching IDs
        if (selectivity <= 0.02 || (!matching_labels.empty() && matching_labels.size() <= 5000)) {
            if (matching_labels.empty() && !metadata_store_.can_use_inverted_index(*filter)) {
                // Collect all matching labels
                matching_labels.reserve(total_elements);
                for (std::size_t i = 0; i < total_elements; ++i) {
                    if (!deleted_[i] && metadata_store_.evaluate(labels_[i], *filter)) {
                        matching_labels.push_back(labels_[i]);
                    }
                }
            }
            return brute_force_search_subset(q, k, matching_labels);
        }

        // Broad filter: scale ef and filter results
        std::size_t base_ef = (ef_search_override > 0) ? ef_search_override : ef_search_.load();
        std::size_t scale = std::min(static_cast<std::size_t>(1.0 / std::max(selectivity, 0.05)), std::size_t(20));
        std::size_t ef = std::max(base_ef * scale, k * 2);

        idx_t curr_obj;
        int curr_max_level;
        {
            std::lock_guard<std::mutex> ep_lock(entry_point_mutex_);
            curr_obj = entry_point_.load();
            curr_max_level = max_level_.load();
        }

        distance_computations_query_++;
        float cur_dist_f = compute_dist_by_idx(q, curr_obj);
        for (int l = curr_max_level; l > 0; --l) {
            bool changed = true;
            std::vector<idx_t> links;
            while (changed) {
                changed = false;
                get_node_links(curr_obj, l, links);
                for (idx_t nbr : links) {
                    distance_computations_query_++;
                    float d = compute_dist_by_idx(q, nbr);
                    if (d < cur_dist_f) {
                        cur_dist_f = d;
                        curr_obj = nbr;
                        changed = true;
                    }
                }
            }
        }

        VisitedListGuard visited_guard(*visited_pool_);
        std::vector<Candidate> candidates = search_layer(q, curr_obj, ef, 0, visited_guard.get(), false);

        SearchResult result;
        result.ids.reserve(k);
        result.dists.reserve(k);

        for (const auto& c : candidates) {
            if (!deleted_[c.idx] && metadata_store_.evaluate(labels_[c.idx], *filter)) {
                result.ids.push_back(labels_[c.idx]);
                result.dists.push_back(c.dist);
                if (result.ids.size() == k) break;
            }
        }

        while (result.ids.size() < k) {
            result.ids.push_back(-1);
            result.dists.push_back(std::numeric_limits<float>::infinity());
        }

        return result;
    }

    // Standard unfiltered search
    std::size_t ef = (ef_search_override > 0) ? ef_search_override : ef_search_.load();
    ef = std::max(ef, k);

    idx_t curr_obj;
    int curr_max_level;
    {
        std::lock_guard<std::mutex> ep_lock(entry_point_mutex_);
        curr_obj = entry_point_.load();
        curr_max_level = max_level_.load();
    }

    distance_computations_query_++;
    float cur_dist = compute_dist_by_idx(q, curr_obj);
    for (int l = curr_max_level; l > 0; --l) {
        bool changed = true;
        std::vector<idx_t> links;
        while (changed) {
            changed = false;
            get_node_links(curr_obj, l, links);
            for (idx_t nbr : links) {
                distance_computations_query_++;
                float d = compute_dist_by_idx(q, nbr);
                if (d < cur_dist) {
                    cur_dist = d;
                    curr_obj = nbr;
                    changed = true;
                }
            }
        }
    }

    VisitedListGuard visited_guard(*visited_pool_);
    std::vector<Candidate> candidates = search_layer(q, curr_obj, ef, 0, visited_guard.get(), false);


    SearchResult result;
    result.ids.reserve(k);
    result.dists.reserve(k);

    for (const auto& c : candidates) {
        if (!deleted_[c.idx]) {
            result.ids.push_back(labels_[c.idx]);
            result.dists.push_back(c.dist);
            if (result.ids.size() == k) break;
        }
    }

    while (result.ids.size() < k) {
        result.ids.push_back(-1);
        result.dists.push_back(std::numeric_limits<float>::infinity());
    }

    return result;
}

std::vector<SearchResult> HNSWIndex::search_batch(const float* queries,
                                                  std::size_t n_queries,
                                                  std::size_t k,
                                                  std::size_t ef_search_override,
                                                  const FilterExpr* filter,
                                                  std::size_t num_threads) const {
    if (n_queries == 0) return {};
    if (num_threads == 0) {
        num_threads = std::max(1u, std::thread::hardware_concurrency());
    }

    std::vector<SearchResult> results(n_queries);
    if (num_threads == 1 || n_queries == 1) {
        for (std::size_t i = 0; i < n_queries; ++i) {
            results[i] = search(queries + i * dim_, k, ef_search_override, filter);
        }
        return results;
    }

    std::vector<std::future<void>> futures;
    std::size_t chunk_size = (n_queries + num_threads - 1) / num_threads;

    for (std::size_t t = 0; t < num_threads; ++t) {
        std::size_t start = t * chunk_size;
        std::size_t end = std::min(start + chunk_size, n_queries);
        if (start >= end) break;

        futures.push_back(std::async(std::launch::async, [this, queries, k, ef_search_override, filter, start, end, &results]() {
            for (std::size_t i = start; i < end; ++i) {
                results[i] = search(queries + i * dim_, k, ef_search_override, filter);
            }
        }));
    }

    for (auto& f : futures) {
        f.get();
    }

    return results;
}

std::size_t HNSWIndex::delete_labels(const std::vector<label_t>& labels) {
    std::unique_lock<std::shared_mutex> lock(structure_lock_);
    std::size_t deleted_count = 0;

    {
        std::lock_guard<std::mutex> map_lock(label_map_mutex_);
        for (label_t label : labels) {
            auto it = label_to_idx_.find(label);
            if (it != label_to_idx_.end()) {
                idx_t idx = it->second;
                if (!deleted_[idx]) {
                    deleted_[idx] = 1;
                    --live_count_;
                    ++deleted_count;
                    metadata_store_.remove_metadata(label);
                }
            }
        }
    }
    return deleted_count;
}

bool HNSWIndex::delete_label(label_t label) {
    return delete_labels({label}) > 0;
}

std::size_t HNSWIndex::size() const {
    return live_count_.load();
}

bool HNSWIndex::contains(label_t label) const {
    std::shared_lock<std::shared_mutex> lock(structure_lock_);
    std::lock_guard<std::mutex> map_lock(label_map_mutex_);
    auto it = label_to_idx_.find(label);
    if (it == label_to_idx_.end()) return false;
    return deleted_[it->second] == 0;
}

std::vector<float> HNSWIndex::get_vector(label_t label) const {
    std::shared_lock<std::shared_mutex> lock(structure_lock_);
    idx_t idx;
    {
        std::lock_guard<std::mutex> map_lock(label_map_mutex_);
        auto it = label_to_idx_.find(label);
        if (it == label_to_idx_.end() || deleted_[it->second]) {
            throw InvalidArgument("Label " + std::to_string(label) + " not found in index");
        }
        idx = it->second;
    }
    const float* ptr = vectors_.data() + idx * dim_;
    return std::vector<float>(ptr, ptr + dim_);
}

void HNSWIndex::get_vector(label_t label, float* out) const {
    std::shared_lock<std::shared_mutex> lock(structure_lock_);
    idx_t idx;
    {
        std::lock_guard<std::mutex> map_lock(label_map_mutex_);
        auto it = label_to_idx_.find(label);
        if (it == label_to_idx_.end() || deleted_[it->second]) {
            throw InvalidArgument("Label " + std::to_string(label) + " not found in index");
        }
        idx = it->second;
    }
    std::memcpy(out, vectors_.data() + idx * dim_, dim_ * sizeof(float));
}

void HNSWIndex::resize(std::size_t new_max_elements) {
    std::unique_lock<std::shared_mutex> lock(structure_lock_);
    if (new_max_elements < cur_element_count_.load()) {
        throw InvalidArgument("Cannot resize to " + std::to_string(new_max_elements) +
                              ", smaller than current elements: " + std::to_string(cur_element_count_.load()));
    }

    max_elements_ = new_max_elements;
    vectors_.resize(max_elements_ * dim_);
    labels_.resize(max_elements_);
    levels_.resize(max_elements_, 0);
    deleted_.resize(max_elements_, 0);
    layer0_links_.resize(max_elements_ * (1 + M0_), 0);
    upper_links_.resize(max_elements_);

    visited_pool_->resize(max_elements_);
}

std::unique_ptr<HNSWIndex> HNSWIndex::compact() const {
    std::unique_lock<std::shared_mutex> lock(structure_lock_);

    std::size_t live = live_count_.load();
    std::size_t new_capacity = std::max(max_elements_, live);

    auto new_idx = std::make_unique<HNSWIndex>(
        dim_, metric_, new_capacity, M_, ef_construction_, ef_search_.load(), seed_);


    std::size_t total = cur_element_count_.load();
    for (std::size_t i = 0; i < total; ++i) {
        if (!deleted_[i]) {
            label_t l = labels_[i];
            MetaMap meta = metadata_store_.get_metadata(l);
            new_idx->add(vectors_.data() + i * dim_, l, meta.empty() ? nullptr : &meta);
        }
    }

    return new_idx;
}

std::unordered_map<std::string, double> HNSWIndex::stats() const {
    std::shared_lock<std::shared_mutex> lock(structure_lock_);

    std::unordered_map<std::string, double> res;
    std::size_t total = cur_element_count_.load();
    std::size_t live = live_count_.load();
    std::size_t deleted = total > live ? (total - live) : 0;
    double tombstone_ratio = total > 0 ? (static_cast<double>(deleted) / total) : 0.0;


    res["count"] = static_cast<double>(live);
    res["size"] = static_cast<double>(live);

    res["capacity"] = static_cast<double>(max_elements_);
    res["total_elements"] = static_cast<double>(total);
    res["deleted"] = static_cast<double>(deleted);
    res["tombstones"] = static_cast<double>(deleted);
    res["tombstone_ratio"] = tombstone_ratio;
    res["needs_compaction"] = (tombstone_ratio > 0.3) ? 1.0 : 0.0;
    res["entry_point"] = static_cast<double>(entry_point_.load());
    res["dim"] = static_cast<double>(dim_);
    res["M"] = static_cast<double>(M_);
    res["ef_construction"] = static_cast<double>(ef_construction_);
    res["ef_search"] = static_cast<double>(ef_search_.load());
    res["max_level"] = static_cast<double>(max_level_.load());


    double total_deg = 0.0;
    for (std::size_t i = 0; i < total; ++i) {
        if (!deleted_[i]) {
            total_deg += *get_layer0_link_count_ptr(static_cast<idx_t>(i));
        }
    }
    res["avg_degree_l0"] = live > 0 ? (total_deg / live) : 0.0;

    std::size_t mem = sizeof(*this) +
                      vectors_.capacity() * sizeof(float) +
                      labels_.capacity() * sizeof(label_t) +
                      levels_.capacity() * sizeof(uint8_t) +
                      deleted_.capacity() * sizeof(uint8_t) +
                      layer0_links_.capacity() * sizeof(uint32_t);
    res["memory_bytes"] = static_cast<double>(mem);

    return res;
}

std::unordered_map<std::string, double> HNSWIndex::diagnostics() const {
    std::shared_lock<std::shared_mutex> lock(structure_lock_);
    std::unordered_map<std::string, double> d;

    std::size_t total = cur_element_count_.load();
    std::size_t live = live_count_.load();
    d["total_elements"] = static_cast<double>(total);
    d["live_elements"] = static_cast<double>(live);

    if (total == 0) {
        d["duplicate_links_count"] = 0.0;
        d["self_links_count"] = 0.0;
        d["unreachable_nodes_count"] = 0.0;
        d["zero_in_degree_count"] = 0.0;
        d["weakly_connected_components"] = 0.0;
        d["reciprocal_links_fraction"] = 1.0;
        d["layer0_degree_mean"] = 0.0;
        d["layer0_degree_min"] = 0.0;
        d["layer0_degree_max"] = 0.0;
        d["cumulative_duplicate_insert_attempts"] = 0.0;
        d["cumulative_heuristic_calls"] = 0.0;
        d["cumulative_overflow_reselections"] = 0.0;
        d["distance_computations_build"] = 0.0;
        d["distance_computations_query"] = 0.0;
        return d;
    }

    uint64_t dup_count = 0;
    uint64_t self_links = 0;
    double min_deg = 1e9, max_deg = 0.0, sum_deg = 0.0;
    std::vector<uint32_t> in_degree(total, 0);
    std::vector<std::vector<uint32_t>> adj(total);

    for (std::size_t i = 0; i < total; ++i) {
        if (deleted_[i]) continue;
        uint32_t cnt = *get_layer0_link_count_ptr(static_cast<idx_t>(i));
        const uint32_t* links = get_layer0_links_ptr(static_cast<idx_t>(i));

        sum_deg += cnt;
        if (cnt < min_deg) min_deg = cnt;
        if (cnt > max_deg) max_deg = cnt;

        std::unordered_set<uint32_t> seen;
        for (uint32_t j = 0; j < cnt; ++j) {
            uint32_t nbr = links[j];
            if (nbr == i) ++self_links;
            if (!seen.insert(nbr).second) {
                ++dup_count;
            }
            if (nbr < total) {
                in_degree[nbr]++;
                adj[i].push_back(nbr);
            }
        }
    }

    // Upper layer dups & self links
    for (std::size_t i = 0; i < total; ++i) {
        if (deleted_[i]) continue;
        int lvl = levels_[i];
        for (int l = 1; l <= lvl; ++l) {
            if (l - 1 < static_cast<int>(upper_links_[i].size())) {
                const auto& u_links = upper_links_[i][l - 1];
                std::unordered_set<uint32_t> u_seen;
                for (uint32_t nbr : u_links) {
                    if (nbr == i) ++self_links;
                    if (!u_seen.insert(nbr).second) {
                        ++dup_count;
                    }
                }
            }
        }
    }

    d["duplicate_links_count"] = static_cast<double>(dup_count);
    d["self_links_count"] = static_cast<double>(self_links);
    d["layer0_degree_mean"] = live > 0 ? (sum_deg / live) : 0.0;
    d["layer0_degree_min"] = (min_deg > 1e8) ? 0.0 : min_deg;
    d["layer0_degree_max"] = max_deg;

    // Zero in-degree nodes (excluding entry point)
    idx_t ep = entry_point_.load();
    uint64_t zero_in = 0;
    for (std::size_t i = 0; i < total; ++i) {
        if (!deleted_[i] && i != ep && in_degree[i] == 0) {
            ++zero_in;
        }
    }
    d["zero_in_degree_count"] = static_cast<double>(zero_in);

    // BFS reachability from entry point at layer 0
    std::vector<bool> reachable(total, false);
    if (ep < total && !deleted_[ep]) {
        std::queue<idx_t> q;
        q.push(ep);
        reachable[ep] = true;
        while (!q.empty()) {
            idx_t curr = q.front();
            q.pop();
            for (uint32_t nbr : adj[curr]) {
                if (nbr < total && !reachable[nbr]) {
                    reachable[nbr] = true;
                    q.push(nbr);
                }
            }
        }
    }

    uint64_t unreachable = 0;
    for (std::size_t i = 0; i < total; ++i) {
        if (!deleted_[i] && !reachable[i]) {
            ++unreachable;
        }
    }
    d["unreachable_nodes_count"] = static_cast<double>(unreachable);

    // Weakly connected components
    std::vector<std::vector<uint32_t>> undir_adj(total);
    for (std::size_t i = 0; i < total; ++i) {
        if (deleted_[i]) continue;
        for (uint32_t nbr : adj[i]) {
            undir_adj[i].push_back(nbr);
            undir_adj[nbr].push_back(static_cast<uint32_t>(i));
        }
    }

    std::vector<bool> visited_wcc(total, false);
    uint64_t num_wcc = 0;
    for (std::size_t i = 0; i < total; ++i) {
        if (deleted_[i] || visited_wcc[i]) continue;
        ++num_wcc;
        std::queue<uint32_t> q;
        q.push(static_cast<uint32_t>(i));
        visited_wcc[i] = true;
        while (!q.empty()) {
            uint32_t u = q.front();
            q.pop();
            for (uint32_t v : undir_adj[u]) {
                if (v < total && !deleted_[v] && !visited_wcc[v]) {
                    visited_wcc[v] = true;
                    q.push(v);
                }
            }
        }
    }
    d["weakly_connected_components"] = static_cast<double>(num_wcc);

    // Reciprocal links fraction
    uint64_t total_edges = 0;
    uint64_t reciprocal_edges = 0;
    for (std::size_t i = 0; i < total; ++i) {
        if (deleted_[i]) continue;
        for (uint32_t nbr : adj[i]) {
            if (nbr >= total || deleted_[nbr]) continue;
            ++total_edges;
            const auto& nbr_links = adj[nbr];
            if (std::find(nbr_links.begin(), nbr_links.end(), static_cast<uint32_t>(i)) != nbr_links.end()) {
                ++reciprocal_edges;
            }
        }
    }
    d["reciprocal_links_fraction"] = total_edges > 0 ? (static_cast<double>(reciprocal_edges) / total_edges) : 1.0;

    // Cumulative counters
    d["cumulative_duplicate_insert_attempts"] = static_cast<double>(duplicate_insert_attempts_.load());
    d["cumulative_heuristic_calls"] = static_cast<double>(heuristic_selection_calls_.load());
    d["cumulative_overflow_reselections"] = static_cast<double>(overflow_reselection_calls_.load());
    d["distance_computations_build"] = static_cast<double>(distance_computations_build_.load());
    d["distance_computations_query"] = static_cast<double>(distance_computations_query_.load());

    return d;
}

bool HNSWIndex::validate_graph() const {
    std::shared_lock<std::shared_mutex> lock(structure_lock_);
    std::size_t total = cur_element_count_.load();
    if (total == 0) return true;

    int cur_max_lvl = max_level_.load();
    idx_t ep = entry_point_.load();

    if (ep >= total) return false;
    if (static_cast<int>(levels_[ep]) != cur_max_lvl) return false;

    for (std::size_t i = 0; i < total; ++i) {
        idx_t idx = static_cast<idx_t>(i);
        uint32_t l0_cnt = *get_layer0_link_count_ptr(idx);
        if (l0_cnt > M0_) return false;

        const uint32_t* l0_links = get_layer0_links_ptr(idx);
        std::vector<uint32_t> seen;
        for (uint32_t j = 0; j < l0_cnt; ++j) {
            uint32_t target = l0_links[j];
            if (target >= total) return false;
            if (target == idx) return false;
            if (std::find(seen.begin(), seen.end(), target) != seen.end()) {
                return false;
            }
            seen.push_back(target);
        }

        int lvl = levels_[idx];
        for (int l = 1; l <= lvl; ++l) {
            if (l - 1 < static_cast<int>(upper_links_[idx].size())) {
                const auto& u_links = upper_links_[idx][l - 1];
                if (u_links.size() > Mmax_) return false;
                std::vector<uint32_t> u_seen;
                for (uint32_t target : u_links) {
                    if (target >= total) return false;
                    if (target == idx) return false;
                    if (std::find(u_seen.begin(), u_seen.end(), target) != u_seen.end()) {
                        return false;
                    }
                    u_seen.push_back(target);
                }
            }
        }
    }

    return true;
}

double HNSWIndex::bench_search_latency(const float* queries, std::size_t n_queries, std::size_t k, std::size_t ef) const {
    if (n_queries == 0) return 0.0;
    std::size_t warmup = std::min(n_queries, std::size_t(500));
    for (std::size_t i = 0; i < warmup; ++i) {
        search(queries + i * dim_, k, ef);
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    for (std::size_t i = 0; i < n_queries; ++i) {
        search(queries + i * dim_, k, ef);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = t1 - t0;
    return elapsed.count() / static_cast<double>(n_queries);
}

void HNSWIndex::save(const std::string& filename) const {
    std::shared_lock<std::shared_mutex> lock(structure_lock_);

    std::vector<uint8_t> buffer;
    const char magic[8] = {'H', 'N', 'S', 'W', 'D', 'B', '0', '1'};
    buffer.insert(buffer.end(), magic, magic + 8);

    uint32_t format_version = 2;
    uint32_t metric_val = static_cast<uint32_t>(metric_);
    uint32_t dim_val = static_cast<uint32_t>(dim_);
    uint64_t count_val = cur_element_count_.load();
    uint64_t max_elements_val = max_elements_;
    uint32_t M_val = static_cast<uint32_t>(M_);
    uint32_t efc_val = static_cast<uint32_t>(ef_construction_);
    uint32_t efs_val = static_cast<uint32_t>(ef_search_.load());
    uint32_t ep_val = entry_point_.load();
    int32_t max_lvl_val = max_level_.load();
    uint64_t seed_val = seed_;

    auto append_raw = [&buffer](const auto& val) {
        const uint8_t* p = reinterpret_cast<const uint8_t*>(&val);
        buffer.insert(buffer.end(), p, p + sizeof(val));
    };

    auto append_section = [&buffer, &append_raw](const void* data, uint64_t bytes) {
        append_raw(bytes);
        if (bytes > 0 && data != nullptr) {
            const uint8_t* p = static_cast<const uint8_t*>(data);
            buffer.insert(buffer.end(), p, p + bytes);
        }
    };

    append_raw(format_version);
    append_raw(metric_val);
    append_raw(dim_val);
    append_raw(count_val);
    append_raw(max_elements_val);
    append_raw(M_val);
    append_raw(efc_val);
    append_raw(efs_val);
    append_raw(ep_val);
    append_raw(max_lvl_val);
    append_raw(seed_val);

    append_section(labels_.data(), count_val * sizeof(label_t));
    append_section(levels_.data(), count_val * sizeof(uint8_t));
    append_section(vectors_.data(), count_val * dim_val * sizeof(float));
    append_section(layer0_links_.data(), count_val * (1 + 2 * M_val) * sizeof(uint32_t));

    std::vector<uint8_t> upper_buf;
    for (std::size_t i = 0; i < count_val; ++i) {
        if (levels_[i] > 0) {
            uint32_t node_idx = static_cast<uint32_t>(i);
            uint32_t num_layers = static_cast<uint32_t>(upper_links_[i].size());

            const uint8_t* p = reinterpret_cast<const uint8_t*>(&node_idx);
            upper_buf.insert(upper_buf.end(), p, p + sizeof(node_idx));
            p = reinterpret_cast<const uint8_t*>(&num_layers);
            upper_buf.insert(upper_buf.end(), p, p + sizeof(num_layers));

            for (uint32_t l = 0; l < num_layers; ++l) {
                uint32_t link_cnt = static_cast<uint32_t>(upper_links_[i][l].size());
                p = reinterpret_cast<const uint8_t*>(&link_cnt);
                upper_buf.insert(upper_buf.end(), p, p + sizeof(link_cnt));
                if (link_cnt > 0) {
                    p = reinterpret_cast<const uint8_t*>(upper_links_[i][l].data());
                    upper_buf.insert(upper_buf.end(), p, p + link_cnt * sizeof(uint32_t));
                }
            }
        }
    }
    append_section(upper_buf.data(), upper_buf.size());
    append_section(deleted_.data(), count_val * sizeof(uint8_t));

    // Section 7: metadata
    std::vector<uint8_t> meta_buf;
    metadata_store_.serialize(meta_buf);
    append_section(meta_buf.data(), meta_buf.size());

    uint32_t crc = CRC32::compute(buffer.data(), buffer.size());
    append_raw(crc);

    atomic_write_file(filename, buffer);
}

std::unique_ptr<HNSWIndex> HNSWIndex::load(const std::string& filename) {
    std::vector<uint8_t> buffer;
    read_file_exact(filename, buffer);

    if (buffer.size() < 60 + 4) {
        throw CorruptFile("File too small to be a valid HNSWIndex: " + filename);
    }

    if (std::memcmp(buffer.data(), "HNSWDB01", 8) != 0) {
        throw CorruptFile("Invalid magic header in file: " + filename);
    }

    std::size_t data_len = buffer.size() - 4;
    uint32_t expected_crc = *reinterpret_cast<const uint32_t*>(buffer.data() + data_len);
    uint32_t actual_crc = CRC32::compute(buffer.data(), data_len);
    if (expected_crc != actual_crc) {
        throw CorruptFile("CRC32 checksum mismatch in file: " + filename);
    }

    std::size_t offset = 8;
    auto read_raw = [&buffer, &offset](auto& val) {
        std::memcpy(&val, buffer.data() + offset, sizeof(val));
        offset += sizeof(val);
    };

    auto read_section = [&buffer, &offset](std::vector<uint8_t>& out) {
        if (offset + sizeof(uint64_t) > buffer.size() - 4) {
            throw CorruptFile("Unexpected end of file while reading section header");
        }
        uint64_t bytes = 0;
        std::memcpy(&bytes, buffer.data() + offset, sizeof(bytes));
        offset += sizeof(bytes);
        if (bytes > buffer.size() - 4 || offset + bytes > buffer.size() - 4) {
            throw CorruptFile("Section length out of file bounds");
        }
        out.resize(bytes);
        if (bytes > 0) {
            std::memcpy(out.data(), buffer.data() + offset, bytes);
            offset += bytes;
        }
    };

    uint32_t format_version = 0;
    uint32_t metric_val = 0;
    uint32_t dim_val = 0;
    uint64_t count_val = 0;
    uint64_t max_elements_val = 0;
    uint32_t M_val = 0;
    uint32_t efc_val = 0;
    uint32_t efs_val = 0;
    uint32_t ep_val = 0;
    int32_t max_lvl_val = 0;
    uint64_t seed_val = 0;

    read_raw(format_version);
    if (format_version != 2) {
        if (format_version == 1) {
            throw UnsupportedVersion("re-save the index with the current version");
        }
        throw UnsupportedVersion("Unsupported index file version " + std::to_string(format_version) + ": re-save the index with the current version");
    }
    read_raw(metric_val);
    read_raw(dim_val);
    read_raw(count_val);
    read_raw(max_elements_val);
    read_raw(M_val);
    read_raw(efc_val);
    read_raw(efs_val);
    read_raw(ep_val);
    read_raw(max_lvl_val);
    read_raw(seed_val);

    if (dim_val < 1 || dim_val > 4096 || metric_val > 2 || M_val < 2 || M_val > 128) {
        throw CorruptFile("Corrupt index header fields in: " + filename);
    }

    Metric m = static_cast<Metric>(metric_val);
    auto index = std::make_unique<HNSWIndex>(dim_val, m, max_elements_val, M_val, efc_val, efs_val, seed_val);
    index->cur_element_count_ = count_val;
    index->entry_point_ = ep_val;
    index->max_level_ = max_lvl_val;

    std::vector<uint8_t> s_labels, s_levels, s_vectors, s_l0_links, s_upper_links, s_deleted, s_metadata;
    read_section(s_labels);
    read_section(s_levels);
    read_section(s_vectors);
    read_section(s_l0_links);
    read_section(s_upper_links);
    read_section(s_deleted);
    read_section(s_metadata);

    if (count_val > 0) {
        std::memcpy(index->labels_.data(), s_labels.data(), count_val * sizeof(label_t));
        std::memcpy(index->levels_.data(), s_levels.data(), count_val * sizeof(uint8_t));
        std::memcpy(index->vectors_.data(), s_vectors.data(), count_val * dim_val * sizeof(float));
        std::memcpy(index->layer0_links_.data(), s_l0_links.data(), count_val * (1 + 2 * M_val) * sizeof(uint32_t));
        std::memcpy(index->deleted_.data(), s_deleted.data(), count_val * sizeof(uint8_t));

        std::size_t u_off = 0;
        while (u_off < s_upper_links.size()) {
            uint32_t node_idx = 0;
            uint32_t num_layers = 0;
            std::memcpy(&node_idx, s_upper_links.data() + u_off, sizeof(node_idx)); u_off += sizeof(node_idx);
            std::memcpy(&num_layers, s_upper_links.data() + u_off, sizeof(num_layers)); u_off += sizeof(num_layers);

            if (node_idx < count_val) {
                index->upper_links_[node_idx].resize(num_layers);
                for (uint32_t l = 0; l < num_layers; ++l) {
                    uint32_t link_cnt = 0;
                    std::memcpy(&link_cnt, s_upper_links.data() + u_off, sizeof(link_cnt)); u_off += sizeof(link_cnt);
                    index->upper_links_[node_idx][l].resize(link_cnt);
                    if (link_cnt > 0) {
                        std::memcpy(index->upper_links_[node_idx][l].data(), s_upper_links.data() + u_off, link_cnt * sizeof(uint32_t));
                        u_off += link_cnt * sizeof(uint32_t);
                    }
                }
            }
        }

        if (!s_metadata.empty()) {
            index->metadata_store_.deserialize(s_metadata.data(), s_metadata.size());
        }

        index->live_count_ = 0;
        for (std::size_t i = 0; i < count_val; ++i) {
            index->label_to_idx_[index->labels_[i]] = static_cast<idx_t>(i);
            if (!index->deleted_[i]) {
                ++index->live_count_;
            }
        }
    }

    return index;
}

std::vector<idx_t> HNSWIndex::test_select_neighbors_heuristic(
    const std::vector<float>& query_vec,
    const std::vector<std::pair<float, idx_t>>& candidates_in,
    std::size_t max_links,
    bool keep_pruned) const {
    std::vector<Candidate> cands;
    cands.reserve(candidates_in.size());
    for (const auto& p : candidates_in) {
        cands.push_back({p.first, p.second});
    }
    return select_neighbors_heuristic(query_vec.data(), cands, max_links, keep_pruned);
}

std::vector<std::pair<float, idx_t>> HNSWIndex::test_search_layer(
    const std::vector<float>& query_vec,
    idx_t entry_point,
    std::size_t ef,
    int layer) const {
    VisitedListGuard visited_guard(*visited_pool_);
    auto cands = search_layer(query_vec.data(), entry_point, ef, layer, visited_guard.get(), false);
    std::vector<std::pair<float, idx_t>> res;
    res.reserve(cands.size());
    for (const auto& c : cands) {
        res.push_back({c.dist, c.idx});
    }
    return res;
}

std::vector<std::vector<idx_t>> HNSWIndex::export_layer0_graph() const {
    std::shared_lock<std::shared_mutex> lock(structure_lock_);
    std::size_t total = cur_element_count_.load();
    std::vector<std::vector<idx_t>> graph(total);
    for (std::size_t i = 0; i < total; ++i) {
        uint32_t cnt = *get_layer0_link_count_ptr(static_cast<idx_t>(i));
        const uint32_t* links = get_layer0_links_ptr(static_cast<idx_t>(i));
        graph[i].assign(links, links + cnt);
    }
    return graph;
}

std::vector<uint32_t> HNSWIndex::get_in_degrees_l0() const {
    return in_degrees_l0_;
}

std::size_t HNSWIndex::get_unreachable_count() const {
    auto d = diagnostics();
    auto it = d.find("unreachable_nodes_count");
    if (it != d.end()) {
        return static_cast<std::size_t>(it->second);
    }
    return 0;
}

} // namespace hnswdb

