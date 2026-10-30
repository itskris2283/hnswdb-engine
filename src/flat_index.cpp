#include "hnswdb/flat_index.h"
#include "hnswdb/distance.h"
#include "hnswdb/types.h"
#include "hnswdb/serialization.h"
#include <algorithm>
#include <cmath>
#include <queue>
#include <thread>
#include <future>
#include <mutex>
#include <cstring>
#include <limits>

namespace hnswdb {

FlatIndex::FlatIndex(std::size_t dim, Metric metric)
    : dim_(dim), metric_(metric) {
    if (dim_ < 1 || dim_ > 4096) {
        throw InvalidArgument("Dimension must be between 1 and 4096, got " + std::to_string(dim_));
    }
    if (metric_ == Metric::L2) {
        dist_func_ = distance::get_l2sq_func();
    } else {
        dist_func_ = distance::get_dot_func();
    }
}

float FlatIndex::compute_distance(const float* a, const float* b) const {
    if (metric_ == Metric::L2) {
        return dist_func_(a, b, dim_);
    } else if (metric_ == Metric::IP) {
        return -dist_func_(a, b, dim_);
    } else { // Cosine
        return 1.0f - dist_func_(a, b, dim_);
    }
}

void FlatIndex::add(const float* vector, label_t label) {
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

    std::unique_lock<std::shared_mutex> lock(mutex_);

    auto it = label_to_idx_.find(label);
    if (it != label_to_idx_.end()) {
        // Upsert: replace vector data at existing index or tombstone and append
        idx_t idx = it->second;
        if (deleted_[idx]) {
            deleted_[idx] = 0;
            ++live_count_;
        }
        std::memcpy(vectors_.data() + idx * dim_, vec_to_store, dim_ * sizeof(float));
    } else {
        idx_t new_idx = static_cast<idx_t>(labels_.size());
        labels_.push_back(label);
        deleted_.push_back(0);
        vectors_.insert(vectors_.end(), vec_to_store, vec_to_store + dim_);
        label_to_idx_[label] = new_idx;
        ++live_count_;
    }
}

void FlatIndex::add_batch(const float* vectors, const label_t* labels, std::size_t n, std::size_t num_threads) {
    if (n == 0) return;
    if (num_threads == 0) {
        num_threads = std::max(1u, std::thread::hardware_concurrency());
    }

    // Pre-validate and normalize in parallel
    std::vector<std::vector<float>> normalized_vectors(n);
    std::vector<std::future<void>> futures;

    std::size_t chunk_size = (n + num_threads - 1) / num_threads;
    for (std::size_t t = 0; t < num_threads; ++t) {
        std::size_t start = t * chunk_size;
        std::size_t end = std::min(start + chunk_size, n);
        if (start >= end) break;

        futures.push_back(std::async(std::launch::async, [this, vectors, labels, start, end, &normalized_vectors]() {
            for (std::size_t i = start; i < end; ++i) {
                if (labels[i] < 0) {
                    throw InvalidArgument("Label must be non-negative, got " + std::to_string(labels[i]));
                }
                const float* v = vectors + i * dim_;
                if (!distance::validate_vector(v, dim_)) {
                    throw InvalidVector("Vector at index " + std::to_string(i) + " contains NaN or Inf");
                }
                if (metric_ == Metric::Cosine) {
                    normalized_vectors[i].resize(dim_);
                    distance::normalize_vector(v, normalized_vectors[i].data(), dim_);
                }
            }
        }));
    }

    for (auto& f : futures) {
        f.get();
    }

    // Add sequentially to update index state
    for (std::size_t i = 0; i < n; ++i) {
        const float* v = (metric_ == Metric::Cosine) ? normalized_vectors[i].data() : (vectors + i * dim_);
        add(v, labels[i]);
    }
}

SearchResult FlatIndex::search(const float* query, std::size_t k) const {
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

    std::shared_lock<std::shared_mutex> lock(mutex_);

    // Min-heap to find top-k smallest distances
    // We want the k smallest distances, so use a max-heap of size k
    struct DistPair {
        float dist;
        label_t label;
        bool operator<(const DistPair& other) const {
            return dist < other.dist; // Max-heap: largest distance on top
        }
    };

    std::priority_queue<DistPair> max_heap;

    std::size_t total_nodes = labels_.size();
    for (std::size_t i = 0; i < total_nodes; ++i) {
        if (deleted_[i]) continue;

        const float* vec = vectors_.data() + i * dim_;
        float d = compute_distance(q, vec);

        if (max_heap.size() < k) {
            max_heap.push({d, labels_[i]});
        } else if (d < max_heap.top().dist) {
            max_heap.pop();
            max_heap.push({d, labels_[i]});
        }
    }

    // Extract sorted results (ascending)
    SearchResult res;
    res.ids.resize(k, -1);
    res.dists.resize(k, std::numeric_limits<float>::infinity());

    std::size_t found_count = max_heap.size();
    for (std::size_t i = found_count; i > 0; --i) {
        auto top = max_heap.top();
        max_heap.pop();
        res.ids[i - 1] = top.label;
        res.dists[i - 1] = top.dist;
    }

    return res;
}

std::vector<SearchResult> FlatIndex::search_batch(const float* queries, std::size_t n_queries, std::size_t k, std::size_t num_threads) const {
    if (n_queries == 0) return {};
    if (num_threads == 0) {
        num_threads = std::max(1u, std::thread::hardware_concurrency());
    }

    std::vector<SearchResult> results(n_queries);
    std::vector<std::future<void>> futures;

    std::size_t chunk_size = (n_queries + num_threads - 1) / num_threads;
    for (std::size_t t = 0; t < num_threads; ++t) {
        std::size_t start = t * chunk_size;
        std::size_t end = std::min(start + chunk_size, n_queries);
        if (start >= end) break;

        futures.push_back(std::async(std::launch::async, [this, queries, k, start, end, &results]() {
            for (std::size_t i = start; i < end; ++i) {
                results[i] = search(queries + i * dim_, k);
            }
        }));
    }

    for (auto& f : futures) {
        f.get();
    }

    return results;
}

std::size_t FlatIndex::delete_labels(const std::vector<label_t>& labels) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    std::size_t count = 0;
    for (label_t label : labels) {
        auto it = label_to_idx_.find(label);
        if (it != label_to_idx_.end()) {
            idx_t idx = it->second;
            if (!deleted_[idx]) {
                deleted_[idx] = 1;
                --live_count_;
                ++count;
            }
        }
    }
    return count;
}

bool FlatIndex::delete_label(label_t label) {
    return delete_labels({label}) > 0;
}

std::size_t FlatIndex::size() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    return live_count_;
}

bool FlatIndex::contains(label_t label) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    auto it = label_to_idx_.find(label);
    if (it == label_to_idx_.end()) return false;
    return deleted_[it->second] == 0;
}

std::vector<float> FlatIndex::get_vector(label_t label) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    auto it = label_to_idx_.find(label);
    if (it == label_to_idx_.end() || deleted_[it->second]) {
        throw InvalidArgument("Label " + std::to_string(label) + " not found in index");
    }
    idx_t idx = it->second;
    const float* ptr = vectors_.data() + idx * dim_;
    return std::vector<float>(ptr, ptr + dim_);
}

void FlatIndex::get_vector(label_t label, float* out) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    auto it = label_to_idx_.find(label);
    if (it == label_to_idx_.end() || deleted_[it->second]) {
        throw InvalidArgument("Label " + std::to_string(label) + " not found in index");
    }
    idx_t idx = it->second;
    std::memcpy(out, vectors_.data() + idx * dim_, dim_ * sizeof(float));
}

void FlatIndex::save(const std::string& filename) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);

    // Binary format for FlatIndex:
    // Magic "FLATDB01" (8 bytes)
    // u32 metric
    // u32 dim
    // u64 total_elements
    // u64 live_elements
    // Section: labels (total_elements * sizeof(label_t))
    // Section: deleted (total_elements * sizeof(uint8_t))
    // Section: vectors (total_elements * dim * sizeof(float))
    // Trailer: u32 CRC32
    std::vector<uint8_t> buffer;
    const char magic[8] = {'F', 'L', 'A', 'T', 'D', 'B', '0', '1'};
    buffer.insert(buffer.end(), magic, magic + 8);

    uint32_t metric_val = static_cast<uint32_t>(metric_);
    uint32_t dim_val = static_cast<uint32_t>(dim_);
    uint64_t total_elements = labels_.size();
    uint64_t live_elements = live_count_;

    auto append_val = [&buffer](const auto& val) {
        const uint8_t* p = reinterpret_cast<const uint8_t*>(&val);
        buffer.insert(buffer.end(), p, p + sizeof(val));
    };

    append_val(metric_val);
    append_val(dim_val);
    append_val(total_elements);
    append_val(live_elements);

    if (total_elements > 0) {
        // Labels
        const uint8_t* lbl_ptr = reinterpret_cast<const uint8_t*>(labels_.data());
        buffer.insert(buffer.end(), lbl_ptr, lbl_ptr + total_elements * sizeof(label_t));

        // Deleted
        buffer.insert(buffer.end(), deleted_.begin(), deleted_.end());

        // Vectors
        const uint8_t* vec_ptr = reinterpret_cast<const uint8_t*>(vectors_.data());
        buffer.insert(buffer.end(), vec_ptr, vec_ptr + total_elements * dim_ * sizeof(float));
    }

    uint32_t crc = CRC32::compute(buffer.data(), buffer.size());
    append_val(crc);

    atomic_write_file(filename, buffer);
}

std::unique_ptr<FlatIndex> FlatIndex::load(const std::string& filename) {
    std::vector<uint8_t> buffer;
    read_file_exact(filename, buffer);

    if (buffer.size() < 8 + 4 + 4 + 8 + 8 + 4) {
        throw CorruptFile("File too small to be a valid FlatIndex: " + filename);
    }

    // Check magic
    if (std::memcmp(buffer.data(), "FLATDB01", 8) != 0) {
        throw CorruptFile("Invalid magic header in file: " + filename);
    }

    // Verify CRC32
    std::size_t data_len = buffer.size() - 4;
    uint32_t expected_crc = *reinterpret_cast<const uint32_t*>(buffer.data() + data_len);
    uint32_t actual_crc = CRC32::compute(buffer.data(), data_len);
    if (expected_crc != actual_crc) {
        throw CorruptFile("CRC32 mismatch in file: " + filename);
    }

    std::size_t offset = 8;
    auto read_val = [&buffer, &offset](auto& val) {
        std::memcpy(&val, buffer.data() + offset, sizeof(val));
        offset += sizeof(val);
    };

    uint32_t metric_val = 0;
    uint32_t dim_val = 0;
    uint64_t total_elements = 0;
    uint64_t live_elements = 0;

    read_val(metric_val);
    read_val(dim_val);
    read_val(total_elements);
    read_val(live_elements);

    if (dim_val < 1 || dim_val > 4096) {
        throw CorruptFile("Invalid dimension in file: " + std::to_string(dim_val));
    }
    if (metric_val > 2) {
        throw CorruptFile("Invalid metric in file: " + std::to_string(metric_val));
    }

    std::size_t expected_bytes = 8 + 4 + 4 + 8 + 8 +
                                 total_elements * sizeof(label_t) +
                                 total_elements * sizeof(uint8_t) +
                                 total_elements * dim_val * sizeof(float) + 4;
    if (buffer.size() != expected_bytes) {
        throw CorruptFile("Unexpected file size in: " + filename);
    }

    Metric m = static_cast<Metric>(metric_val);
    auto index = std::make_unique<FlatIndex>(dim_val, m);

    if (total_elements > 0) {
        index->labels_.resize(total_elements);
        std::memcpy(index->labels_.data(), buffer.data() + offset, total_elements * sizeof(label_t));
        offset += total_elements * sizeof(label_t);

        index->deleted_.resize(total_elements);
        std::memcpy(index->deleted_.data(), buffer.data() + offset, total_elements * sizeof(uint8_t));
        offset += total_elements * sizeof(uint8_t);

        index->vectors_.resize(total_elements * dim_val);
        std::memcpy(index->vectors_.data(), buffer.data() + offset, total_elements * dim_val * sizeof(float));
        offset += total_elements * dim_val * sizeof(float);

        index->live_count_ = 0;
        for (std::size_t i = 0; i < total_elements; ++i) {
            index->label_to_idx_[index->labels_[i]] = static_cast<idx_t>(i);
            if (!index->deleted_[i]) {
                ++index->live_count_;
            }
        }
    }

    return index;
}

} // namespace hnswdb
