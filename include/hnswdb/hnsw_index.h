#pragma once
#include "types.h"
#include "distance.h"
#include "visited_pool.h"
#include "metadata.h"
#include <vector>
#include <string>
#include <unordered_map>
#include <memory>
#include <shared_mutex>
#include <mutex>
#include <atomic>
#include <random>
#include <cstddef>

namespace hnswdb {

// Simple lightweight spinlock for link list stripes
class SpinLock {
public:
    void lock() {
        while (flag_.test_and_set(std::memory_order_acquire)) {
#if defined(__x86_64__) || defined(_M_X64)
#if defined(_MSC_VER)
            _mm_pause();
#elif defined(__GNUC__) || defined(__clang__)
            __builtin_ia32_pause();
#endif
#endif
        }
    }

    void unlock() {
        flag_.clear(std::memory_order_release);
    }

private:
    std::atomic_flag flag_ = ATOMIC_FLAG_INIT;
};

class SpinLockGuard {
public:
    explicit SpinLockGuard(SpinLock& lock) : lock_(lock) {
        lock_.lock();
    }
    ~SpinLockGuard() {
        lock_.unlock();
    }
private:
    SpinLock& lock_;
};

struct BuildOptions {
    std::string new_node_degree = "M";  // "M" (adopted default) vs "M0"
    bool keep_pruned = false;           // false (adopted default) vs true
    bool protect_last_in_edge = false;  // false (default) vs true
    std::string overflow_selection = "heuristic"; // "heuristic" (default) vs "closest"
    std::string layer_entry = "closest"; // "closest" (default) vs "all"
};

class HNSWIndex {
public:
    HNSWIndex(std::size_t dim,
              Metric metric,
              std::size_t max_elements,
              std::size_t M = 16,
              std::size_t ef_construction = 200,
              std::size_t ef_search = 50,
              uint64_t seed = 42,
              BuildOptions build_options = BuildOptions{});

    ~HNSWIndex() = default;

    // Non-copyable, movable
    HNSWIndex(const HNSWIndex&) = delete;
    HNSWIndex& operator=(const HNSWIndex&) = delete;
    HNSWIndex(HNSWIndex&&) noexcept;
    HNSWIndex& operator=(HNSWIndex&&) noexcept;

    // Core operations
    void add(const float* vector, label_t label, const MetaMap* metadata = nullptr);
    void add_batch(const float* vectors,
                   const label_t* labels,
                   std::size_t n,
                   const std::vector<MetaMap>* metadata = nullptr,
                   std::size_t num_threads = 0);

    SearchResult search(const float* query,
                        std::size_t k,
                        std::size_t ef_search_override = 0,
                        const FilterExpr* filter = nullptr) const;

    std::vector<SearchResult> search_batch(const float* queries,
                                           std::size_t n_queries,
                                           std::size_t k,
                                           std::size_t ef_search_override = 0,
                                           const FilterExpr* filter = nullptr,
                                           std::size_t num_threads = 0) const;

    std::size_t delete_labels(const std::vector<label_t>& labels);
    bool delete_label(label_t label);

    // Metadata management
    void set_metadata(label_t label, const MetaMap& metadata);
    MetaMap get_metadata(label_t label) const;
    bool has_metadata(label_t label) const;

    // Persistence
    void save(const std::string& filename) const;
    static std::unique_ptr<HNSWIndex> load(const std::string& filename);

    // Index management
    void resize(std::size_t new_max_elements);
    std::unique_ptr<HNSWIndex> compact() const;

    // Inspection and metrics
    std::unordered_map<std::string, double> stats() const;
    std::unordered_map<std::string, double> diagnostics() const;
    bool validate_graph() const;

    void set_ef_search(std::size_t ef) { ef_search_ = ef; }
    std::size_t get_ef_search() const { return ef_search_; }

    std::size_t size() const;
    std::size_t capacity() const { return max_elements_; }
    std::size_t dim() const { return dim_; }
    Metric metric() const { return metric_; }
    std::size_t M() const { return M_; }
    std::size_t ef_construction() const { return ef_construction_; }

    bool contains(label_t label) const;
    std::vector<float> get_vector(label_t label) const;
    void get_vector(label_t label, float* out) const;

    double bench_search_latency(const float* queries, std::size_t n_queries, std::size_t k, std::size_t ef = 0) const;

    int get_random_level();
    float compute_dist(const float* a, const float* b) const;
    float compute_dist_by_idx(const float* query, idx_t idx) const;

    void set_build_options(const BuildOptions& options) { build_options_ = options; }
    const BuildOptions& get_build_options() const { return build_options_; }

    // Test and diagnostic hooks
    std::vector<idx_t> test_select_neighbors_heuristic(
        const std::vector<float>& query_vec,
        const std::vector<std::pair<float, idx_t>>& candidates,
        std::size_t max_links,
        bool keep_pruned) const;

    std::vector<std::pair<float, idx_t>> test_search_layer(
        const std::vector<float>& query_vec,
        idx_t entry_point,
        std::size_t ef,
        int layer) const;

    std::vector<std::vector<idx_t>> export_layer0_graph() const;
    std::vector<uint32_t> get_in_degrees_l0() const;
    std::size_t get_unreachable_count() const;

private:
    struct Candidate {
        float dist;
        idx_t idx;

        bool operator<(const Candidate& other) const {
            return dist < other.dist;
        }
        bool operator>(const Candidate& other) const {
            return dist > other.dist;
        }
    };

    // Heuristic neighbor selection (Algorithm 4)
    std::vector<idx_t> select_neighbors_heuristic(
        const float* query_vec,
        std::vector<Candidate>& candidates,
        std::size_t max_links,
        bool keep_pruned = true) const;

    // Layer search for graph construction and query
    std::vector<Candidate> search_layer(
        const float* query_vec,
        idx_t entry_point,
        std::size_t ef,
        int layer,
        VisitedList* visited,
        bool is_build = false) const;

    std::vector<Candidate> search_layer_multi(
        const float* query_vec,
        const std::vector<idx_t>& entry_points,
        std::size_t ef,
        int layer,
        VisitedList* visited,
        bool is_build = false) const;


    // Filtered search helpers
    double estimate_selectivity(const FilterExpr& filter, std::vector<label_t>& out_matching) const;
    SearchResult brute_force_search_subset(const float* query, std::size_t k, const std::vector<label_t>& subset) const;

    // Links storage accessors
    uint32_t* get_layer0_link_count_ptr(idx_t idx);
    const uint32_t* get_layer0_link_count_ptr(idx_t idx) const;
    uint32_t* get_layer0_links_ptr(idx_t idx);
    const uint32_t* get_layer0_links_ptr(idx_t idx) const;

    void get_node_links(idx_t idx, int layer, std::vector<idx_t>& out_links) const;
    void set_node_links(idx_t idx, int layer, const std::vector<idx_t>& links);

    SpinLock& get_stripe_lock(idx_t idx) const {
        return striped_locks_[idx % NUM_STRIPES];
    }

    // Parameters
    std::size_t dim_;
    Metric metric_;
    std::size_t max_elements_;
    std::size_t M_;
    std::size_t M0_;
    std::size_t Mmax_;
    std::size_t ef_construction_;
    std::atomic<std::size_t> ef_search_;
    double mL_;
    uint64_t seed_;
    distance::DistFunc dist_func_;

    // Core data structures
    std::vector<float> vectors_;               // [max_elements * dim]
    std::vector<label_t> labels_;             // [max_elements]
    std::vector<uint8_t> levels_;             // [max_elements]
    std::vector<uint8_t> deleted_;            // [max_elements], 1 = deleted
    std::vector<uint32_t> layer0_links_;      // [max_elements * (1 + M0)]
    std::vector<std::vector<std::vector<uint32_t>>> upper_links_; // upper_links_[idx][layer - 1]

    std::unordered_map<label_t, idx_t> label_to_idx_;
    mutable std::mutex label_map_mutex_;
    std::atomic<std::size_t> cur_element_count_{0};
    std::atomic<std::size_t> live_count_{0};

    // Metadata store
    MetadataStore metadata_store_;

    // Entry point
    std::atomic<idx_t> entry_point_{0};
    std::atomic<int> max_level_{-1};
    mutable std::mutex entry_point_mutex_;

    // Random generator
    std::mt19937_64 rng_;
    std::uniform_real_distribution<double> uniform_dist_{0.0, 1.0};
    mutable std::mutex rng_mutex_;

    // Visited pool
    mutable std::unique_ptr<VisitedListPool> visited_pool_;

    // Concurrency control
    static constexpr std::size_t NUM_STRIPES = 1024;
    mutable std::unique_ptr<SpinLock[]> striped_locks_;
    mutable std::shared_mutex structure_lock_;

    // Cumulative diagnostic counters
    mutable std::atomic<uint64_t> duplicate_insert_attempts_{0};
    mutable std::atomic<uint64_t> heuristic_selection_calls_{0};
    mutable std::atomic<uint64_t> overflow_reselection_calls_{0};
    mutable std::atomic<uint64_t> distance_computations_build_{0};
    mutable std::atomic<uint64_t> distance_computations_query_{0};

    // Build configuration & Layer 0 in-degree tracking
    BuildOptions build_options_;
    std::vector<uint32_t> in_degrees_l0_;
};

} // namespace hnswdb
