#pragma once
#include <cstdint>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>
#include <limits>

namespace hnswdb {

using label_t = int64_t;
using idx_t = uint32_t;

enum class Metric {
    L2,
    IP,
    Cosine
};

inline std::string metric_to_string(Metric m) {
    switch (m) {
        case Metric::L2: return "l2";
        case Metric::IP: return "ip";
        case Metric::Cosine: return "cosine";
    }
    return "unknown";
}

inline Metric string_to_metric(const std::string& str) {
    if (str == "l2" || str == "L2" || str == "euclidean") return Metric::L2;
    if (str == "ip" || str == "IP" || str == "inner_product" || str == "dot") return Metric::IP;
    if (str == "cosine" || str == "Cosine" || str == "angular") return Metric::Cosine;
    throw std::invalid_argument("Unknown metric: " + str);
}

struct SearchResult {
    std::vector<label_t> ids;
    std::vector<float> dists;
};

// Exception hierarchy
class HnswdbError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class DimensionMismatch : public HnswdbError {
public:
    using HnswdbError::HnswdbError;
};

class InvalidVector : public HnswdbError {
public:
    using HnswdbError::HnswdbError;
};

class CapacityExceeded : public HnswdbError {
public:
    using HnswdbError::HnswdbError;
};

class CorruptFile : public HnswdbError {
public:
    using HnswdbError::HnswdbError;
};

class UnsupportedVersion : public HnswdbError {
public:
    using HnswdbError::HnswdbError;
};

class InvalidArgument : public HnswdbError {
public:
    using HnswdbError::HnswdbError;
};

} // namespace hnswdb
