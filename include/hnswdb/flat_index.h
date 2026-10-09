#pragma once
#include "types.h"
#include "distance.h"
#include <vector>
#include <string>
#include <unordered_map>
#include <memory>
#include <shared_mutex>
#include <cstddef>

namespace hnswdb {

class FlatIndex {
public:
    FlatIndex(std::size_t dim, Metric metric);
    ~FlatIndex() = default;

    // Non-copyable, movable
    FlatIndex(const FlatIndex&) = delete;
    FlatIndex& operator=(const FlatIndex&) = delete;
    FlatIndex(FlatIndex&&) noexcept = default;
    FlatIndex& operator=(FlatIndex&&) noexcept = default;

    void add(const float* vector, label_t label);
    void add_batch(const float* vectors, const label_t* labels, std::size_t n, std::size_t num_threads = 0);

    SearchResult search(const float* query, std::size_t k) const;
    std::vector<SearchResult> search_batch(const float* queries, std::size_t n_queries, std::size_t k, std::size_t num_threads = 0) const;

    std::size_t delete_labels(const std::vector<label_t>& labels);
    bool delete_label(label_t label);

    void save(const std::string& filename) const;
    static std::unique_ptr<FlatIndex> load(const std::string& filename);

    std::size_t size() const;
    std::size_t dim() const { return dim_; }
    Metric metric() const { return metric_; }

    bool contains(label_t label) const;
    std::vector<float> get_vector(label_t label) const;
    void get_vector(label_t label, float* out) const;

private:
    float compute_distance(const float* a, const float* b) const;

    std::size_t dim_;
    Metric metric_;
    distance::DistFunc dist_func_;

    // Data storage (64-byte aligned vectors buffer)
    std::vector<float> vectors_;
    std::vector<label_t> labels_;
    std::vector<uint8_t> deleted_; // 1 = deleted, 0 = active
    std::unordered_map<label_t, idx_t> label_to_idx_;
    std::size_t live_count_{0};

    mutable std::shared_mutex mutex_;
};

} // namespace hnswdb
