#include <gtest/gtest.h>
#include "hnswdb/hnsw_index.h"
#include "hnswdb/types.h"
#include <random>
#include <vector>
#include <algorithm>

namespace {

std::vector<float> make_random_matrix(std::size_t n, std::size_t d, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> data(n * d);
    for (std::size_t i = 0; i < n * d; ++i) {
        data[i] = dist(rng);
    }
    return data;
}

} // namespace

TEST(HnswDelete, DeletedLabelsNeverReturned) {
    constexpr std::size_t d = 16;
    constexpr std::size_t n = 100;
    constexpr std::size_t k = 10;

    hnswdb::HNSWIndex idx(d, hnswdb::Metric::L2, n, 16, 200, 50, 42);
    auto data = make_random_matrix(n, d, 1);

    for (std::size_t i = 0; i < n; ++i) {
        idx.add(data.data() + i * d, static_cast<hnswdb::label_t>(i));
    }
    EXPECT_EQ(idx.size(), n);

    // Delete 20 labels
    std::vector<hnswdb::label_t> to_delete;
    for (hnswdb::label_t l = 10; l < 30; ++l) {
        to_delete.push_back(l);
    }
    std::size_t deleted_cnt = idx.delete_labels(to_delete);
    EXPECT_EQ(deleted_cnt, 20);
    EXPECT_EQ(idx.size(), n - 20);

    // Search across queries and verify none of the deleted labels are ever returned
    auto queries = make_random_matrix(30, d, 99);
    for (std::size_t q = 0; q < 30; ++q) {
        auto res = idx.search(queries.data() + q * d, k);
        for (hnswdb::label_t id : res.ids) {
            if (id != -1) {
                EXPECT_TRUE(id < 10 || id >= 30) << "Deleted label " << id << " was returned in search!";
            }
        }
    }
}

TEST(HnswDelete, DeleteNonexistentReturnsZero) {
    constexpr std::size_t d = 8;
    hnswdb::HNSWIndex idx(d, hnswdb::Metric::L2, 10);

    std::vector<float> v = {1, 2, 3, 4, 5, 6, 7, 8};
    idx.add(v.data(), 1);

    EXPECT_EQ(idx.delete_labels({999, 1000}), 0);
    EXPECT_EQ(idx.size(), 1);
    EXPECT_TRUE(idx.contains(1));
}

TEST(HnswDelete, DeleteAllThenSearchReturnsNegativeOnes) {
    constexpr std::size_t d = 8;
    constexpr std::size_t n = 20;

    hnswdb::HNSWIndex idx(d, hnswdb::Metric::L2, n);
    auto data = make_random_matrix(n, d, 77);

    std::vector<hnswdb::label_t> all_labels;
    for (std::size_t i = 0; i < n; ++i) {
        idx.add(data.data() + i * d, static_cast<hnswdb::label_t>(i));
        all_labels.push_back(static_cast<hnswdb::label_t>(i));
    }
    EXPECT_EQ(idx.size(), n);

    idx.delete_labels(all_labels);
    EXPECT_EQ(idx.size(), 0);

    std::vector<float> q = {0, 0, 0, 0, 0, 0, 0, 0};
    auto res = idx.search(q.data(), 5);
    EXPECT_EQ(res.ids.size(), 5);
    for (std::size_t i = 0; i < 5; ++i) {
        EXPECT_EQ(res.ids[i], -1);
        EXPECT_TRUE(std::isinf(res.dists[i]));
    }
}

TEST(HnswDelete, ReAddAfterDelete) {
    constexpr std::size_t d = 4;
    hnswdb::HNSWIndex idx(d, hnswdb::Metric::L2, 10);

    std::vector<float> v0 = {1, 1, 1, 1};
    idx.add(v0.data(), 42);
    EXPECT_TRUE(idx.contains(42));
    EXPECT_EQ(idx.size(), 1);

    idx.delete_labels({42});
    EXPECT_FALSE(idx.contains(42));
    EXPECT_EQ(idx.size(), 0);

    std::vector<float> v1 = {2, 2, 2, 2};
    idx.add(v1.data(), 42);
    EXPECT_TRUE(idx.contains(42));
    EXPECT_EQ(idx.size(), 1);

    auto vec = idx.get_vector(42);
    EXPECT_FLOAT_EQ(vec[0], 2.0f);
}

TEST(HnswDelete, EntryPointDeletedSearchStillWorks) {
    constexpr std::size_t d = 8;
    constexpr std::size_t n = 50;

    hnswdb::HNSWIndex idx(d, hnswdb::Metric::L2, n, 16, 200, 50, 42);
    auto data = make_random_matrix(n, d, 33);

    for (std::size_t i = 0; i < n; ++i) {
        idx.add(data.data() + i * d, static_cast<hnswdb::label_t>(i));
    }

    // Delete node 0 (the initial entry point)
    idx.delete_labels({0});
    EXPECT_FALSE(idx.contains(0));
    EXPECT_EQ(idx.size(), n - 1);

    // Search should traverse through node 0 seamlessly and return live nodes
    std::vector<float> q(data.begin(), data.begin() + d); // Exact match for node 0
    auto res = idx.search(q.data(), 5);
    EXPECT_NE(res.ids[0], 0);
    EXPECT_TRUE(idx.contains(res.ids[0]));
}

TEST(HnswDelete, CompactionPreservesLiveSet) {
    constexpr std::size_t d = 16;
    constexpr std::size_t n = 200;

    hnswdb::HNSWIndex idx(d, hnswdb::Metric::L2, n, 16, 200, 50, 42);
    auto data = make_random_matrix(n, d, 55);

    for (std::size_t i = 0; i < n; ++i) {
        idx.add(data.data() + i * d, static_cast<hnswdb::label_t>(i));
    }

    // Delete half the elements (even labels)
    std::vector<hnswdb::label_t> evens;
    for (std::size_t i = 0; i < n; i += 2) {
        evens.push_back(static_cast<hnswdb::label_t>(i));
    }
    idx.delete_labels(evens);
    EXPECT_EQ(idx.size(), n / 2);

    auto st_before = idx.stats();
    EXPECT_EQ(st_before["tombstone_ratio"], 0.5);
    EXPECT_EQ(st_before["needs_compaction"], 1.0);

    // Compact
    auto compacted = idx.compact();
    ASSERT_NE(compacted, nullptr);
    EXPECT_EQ(compacted->size(), n / 2);

    auto st_after = compacted->stats();
    EXPECT_EQ(st_after["tombstone_ratio"], 0.0);
    EXPECT_EQ(st_after["needs_compaction"], 0.0);

    // Verify all odd labels exist and all even labels do not
    for (std::size_t i = 0; i < n; ++i) {
        if (i % 2 == 0) {
            EXPECT_FALSE(compacted->contains(static_cast<hnswdb::label_t>(i)));
        } else {
            EXPECT_TRUE(compacted->contains(static_cast<hnswdb::label_t>(i)));
            auto vec = compacted->get_vector(static_cast<hnswdb::label_t>(i));
            EXPECT_NEAR(vec[0], data[i * d], 1e-5f);
        }
    }
}
