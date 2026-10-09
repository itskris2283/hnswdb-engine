#include <gtest/gtest.h>
#include "hnswdb/hnsw_index.h"
#include "hnswdb/flat_index.h"
#include "hnswdb/types.h"
#include <vector>
#include <limits>

TEST(ErrorHandling, InvalidDimension) {
    EXPECT_THROW(hnswdb::HNSWIndex(0, hnswdb::Metric::L2, 100), hnswdb::InvalidArgument);
    EXPECT_THROW(hnswdb::HNSWIndex(4097, hnswdb::Metric::L2, 100), hnswdb::InvalidArgument);

    EXPECT_THROW(hnswdb::FlatIndex(0, hnswdb::Metric::L2), hnswdb::InvalidArgument);
    EXPECT_THROW(hnswdb::FlatIndex(4097, hnswdb::Metric::L2), hnswdb::InvalidArgument);
}

TEST(ErrorHandling, InvalidM) {
    EXPECT_THROW(hnswdb::HNSWIndex(16, hnswdb::Metric::L2, 100, /*M=*/1), hnswdb::InvalidArgument);
    EXPECT_THROW(hnswdb::HNSWIndex(16, hnswdb::Metric::L2, 100, /*M=*/129), hnswdb::InvalidArgument);
}

TEST(ErrorHandling, InvalidK) {
    hnswdb::HNSWIndex idx(4, hnswdb::Metric::L2, 10);
    std::vector<float> v = {1, 2, 3, 4};
    idx.add(v.data(), 1);

    EXPECT_THROW(idx.search(v.data(), 0), hnswdb::InvalidArgument);

    hnswdb::FlatIndex flat(4, hnswdb::Metric::L2);
    flat.add(v.data(), 1);
    EXPECT_THROW(flat.search(v.data(), 0), hnswdb::InvalidArgument);
}

TEST(ErrorHandling, NegativeLabel) {
    hnswdb::HNSWIndex idx(4, hnswdb::Metric::L2, 10);
    std::vector<float> v = {1, 2, 3, 4};
    EXPECT_THROW(idx.add(v.data(), -1), hnswdb::InvalidArgument);
    EXPECT_THROW(idx.add(v.data(), -100), hnswdb::InvalidArgument);

    hnswdb::FlatIndex flat(4, hnswdb::Metric::L2);
    EXPECT_THROW(flat.add(v.data(), -1), hnswdb::InvalidArgument);
}

TEST(ErrorHandling, NanAndInfVectors) {
    hnswdb::HNSWIndex idx(4, hnswdb::Metric::L2, 10);

    std::vector<float> nan_v = {1.0f, std::numeric_limits<float>::quiet_NaN(), 3.0f, 4.0f};
    EXPECT_THROW(idx.add(nan_v.data(), 1), hnswdb::InvalidVector);
    EXPECT_THROW(idx.search(nan_v.data(), 5), hnswdb::InvalidVector);

    std::vector<float> inf_v = {1.0f, std::numeric_limits<float>::infinity(), 3.0f, 4.0f};
    EXPECT_THROW(idx.add(inf_v.data(), 1), hnswdb::InvalidVector);
    EXPECT_THROW(idx.search(inf_v.data(), 5), hnswdb::InvalidVector);
}

TEST(ErrorHandling, ZeroVectorCosine) {
    hnswdb::HNSWIndex idx(4, hnswdb::Metric::Cosine, 10);
    std::vector<float> zero_v = {0.0f, 0.0f, 0.0f, 0.0f};

    EXPECT_THROW(idx.add(zero_v.data(), 1), hnswdb::InvalidVector);

    std::vector<float> valid_v = {1.0f, 1.0f, 1.0f, 1.0f};
    idx.add(valid_v.data(), 1);
    EXPECT_THROW(idx.search(zero_v.data(), 1), hnswdb::InvalidVector);
}

TEST(ErrorHandling, CapacityExceeded) {
    constexpr std::size_t cap = 5;
    hnswdb::HNSWIndex idx(4, hnswdb::Metric::L2, cap);

    for (std::size_t i = 0; i < cap; ++i) {
        std::vector<float> v = {static_cast<float>(i), 0, 0, 0};
        idx.add(v.data(), static_cast<hnswdb::label_t>(i));
    }
    EXPECT_EQ(idx.size(), cap);

    std::vector<float> extra_v = {100, 0, 0, 0};
    EXPECT_THROW(idx.add(extra_v.data(), 999), hnswdb::CapacityExceeded);
}
