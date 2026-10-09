#include <gtest/gtest.h>
#include "hnswdb/hnsw_index.h"
#include "hnswdb/flat_index.h"
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

TEST(HnswBasic, SmallIndexIsExactAtHighEf) {
    constexpr std::size_t d = 16;
    constexpr std::size_t n = 100;
    constexpr std::size_t k = 10;

    hnswdb::HNSWIndex idx(d, hnswdb::Metric::L2, n, /*M=*/16, /*efc=*/200, /*efs=*/50, /*seed=*/1);
    hnswdb::FlatIndex flat(d, hnswdb::Metric::L2);

    auto data = make_random_matrix(n, d, 7);
    for (std::size_t i = 0; i < n; ++i) {
        idx.add(data.data() + i * d, static_cast<hnswdb::label_t>(i));
        flat.add(data.data() + i * d, static_cast<hnswdb::label_t>(i));
    }
    EXPECT_EQ(idx.size(), n);

    auto q = make_random_matrix(20, d, 99);
    for (std::size_t qi = 0; qi < 20; ++qi) {
        auto got = idx.search(q.data() + qi * d, k, /*ef_search_override=*/500);
        auto want = flat.search(q.data() + qi * d, k);

        EXPECT_EQ(got.ids, want.ids);
        for (std::size_t i = 0; i < k; ++i) {
            EXPECT_NEAR(got.dists[i], want.dists[i], 1e-4f);
        }
    }
}

TEST(HnswBasic, SingleElementIndex) {
    constexpr std::size_t d = 8;
    hnswdb::HNSWIndex idx(d, hnswdb::Metric::L2, 10, 16, 200, 50, 42);

    std::vector<float> vec = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    idx.add(vec.data(), 100);
    EXPECT_EQ(idx.size(), 1);
    EXPECT_TRUE(idx.contains(100));

    std::vector<float> q = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    auto res = idx.search(q.data(), 1);
    EXPECT_EQ(res.ids.size(), 1);
    EXPECT_EQ(res.ids[0], 100);
    EXPECT_NEAR(res.dists[0], 0.0f, 1e-5f);

    // Search k = 3 when only 1 element exists
    auto res3 = idx.search(q.data(), 3);
    EXPECT_EQ(res3.ids.size(), 3);
    EXPECT_EQ(res3.ids[0], 100);
    EXPECT_EQ(res3.ids[1], -1);
    EXPECT_EQ(res3.ids[2], -1);
    EXPECT_TRUE(std::isinf(res3.dists[1]));
    EXPECT_TRUE(std::isinf(res3.dists[2]));
}

TEST(HnswBasic, LabelLookupAndVectorRoundTrip) {
    constexpr std::size_t d = 4;
    hnswdb::HNSWIndex idx(d, hnswdb::Metric::L2, 10);

    std::vector<float> v0 = {1.1f, 2.2f, 3.3f, 4.4f};
    std::vector<float> v1 = {5.5f, 6.6f, 7.7f, 8.8f};

    idx.add(v0.data(), 10);
    idx.add(v1.data(), 20);

    EXPECT_TRUE(idx.contains(10));
    EXPECT_TRUE(idx.contains(20));
    EXPECT_FALSE(idx.contains(30));

    auto ret_v0 = idx.get_vector(10);
    EXPECT_EQ(ret_v0.size(), d);
    for (std::size_t i = 0; i < d; ++i) {
        EXPECT_FLOAT_EQ(ret_v0[i], v0[i]);
    }

    auto ret_v1 = idx.get_vector(20);
    for (std::size_t i = 0; i < d; ++i) {
        EXPECT_FLOAT_EQ(ret_v1[i], v1[i]);
    }

    EXPECT_THROW(idx.get_vector(999), hnswdb::InvalidArgument);
}

TEST(HnswBasic, UpsertReplacesVector) {
    constexpr std::size_t d = 4;
    hnswdb::HNSWIndex idx(d, hnswdb::Metric::L2, 10);

    std::vector<float> v_old = {0.0f, 0.0f, 0.0f, 0.0f};
    std::vector<float> v_new = {10.0f, 10.0f, 10.0f, 10.0f};

    idx.add(v_old.data(), 1);
    EXPECT_EQ(idx.size(), 1);

    auto ret = idx.get_vector(1);
    EXPECT_FLOAT_EQ(ret[0], 0.0f);

    // Upsert
    idx.add(v_new.data(), 1);
    EXPECT_EQ(idx.size(), 1);

    ret = idx.get_vector(1);
    EXPECT_FLOAT_EQ(ret[0], 10.0f);

    std::vector<float> q = {10.0f, 10.0f, 10.0f, 10.0f};
    auto search_res = idx.search(q.data(), 1);
    EXPECT_EQ(search_res.ids[0], 1);
    EXPECT_NEAR(search_res.dists[0], 0.0f, 1e-5f);
}
