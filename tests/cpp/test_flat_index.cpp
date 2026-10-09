#include <gtest/gtest.h>
#include "hnswdb/flat_index.h"
#include "hnswdb/types.h"
#include <random>
#include <vector>
#include <algorithm>
#include <filesystem>

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

TEST(FlatIndexTest, ExactMatchVsBruteForceSortL2) {
    constexpr std::size_t n = 200;
    constexpr std::size_t d = 16;
    constexpr std::size_t k = 10;

    auto data = make_random_matrix(n, d, 42);
    auto queries = make_random_matrix(20, d, 99);

    hnswdb::FlatIndex index(d, hnswdb::Metric::L2);
    for (std::size_t i = 0; i < n; ++i) {
        index.add(data.data() + i * d, static_cast<hnswdb::label_t>(i));
    }
    EXPECT_EQ(index.size(), n);

    for (std::size_t q = 0; q < 20; ++q) {
        const float* query_vec = queries.data() + q * d;
        auto result = index.search(query_vec, k);
        EXPECT_EQ(result.ids.size(), k);
        EXPECT_EQ(result.dists.size(), k);

        // Ground truth
        std::vector<std::pair<float, hnswdb::label_t>> gt;
        for (std::size_t i = 0; i < n; ++i) {
            float dist = 0.0f;
            for (std::size_t dim = 0; dim < d; ++dim) {
                float diff = query_vec[dim] - data[i * d + dim];
                dist += diff * diff;
            }
            gt.push_back({dist, static_cast<hnswdb::label_t>(i)});
        }
        std::sort(gt.begin(), gt.end());

        for (std::size_t i = 0; i < k; ++i) {
            EXPECT_EQ(result.ids[i], gt[i].second) << "Mismatch at rank " << i;
            EXPECT_NEAR(result.dists[i], gt[i].first, 1e-4f);
        }
    }
}

TEST(FlatIndexTest, ExactMatchCosine) {
    constexpr std::size_t n = 100;
    constexpr std::size_t d = 32;
    constexpr std::size_t k = 5;

    auto data = make_random_matrix(n, d, 123);
    auto queries = make_random_matrix(10, d, 456);

    hnswdb::FlatIndex index(d, hnswdb::Metric::Cosine);
    for (std::size_t i = 0; i < n; ++i) {
        index.add(data.data() + i * d, static_cast<hnswdb::label_t>(i));
    }

    for (std::size_t q = 0; q < 10; ++q) {
        const float* query_vec = queries.data() + q * d;
        auto result = index.search(query_vec, k);
        EXPECT_EQ(result.ids.size(), k);

        // Check ascending order of distances
        for (std::size_t i = 1; i < k; ++i) {
            EXPECT_GE(result.dists[i], result.dists[i - 1]);
        }
    }
}

TEST(FlatIndexTest, KGreaterThanNAndEmptyIndex) {
    constexpr std::size_t d = 8;
    hnswdb::FlatIndex index(d, hnswdb::Metric::L2);

    // Empty index search
    std::vector<float> query(d, 0.5f);
    auto empty_res = index.search(query.data(), 5);
    EXPECT_EQ(empty_res.ids.size(), 5);
    for (std::size_t i = 0; i < 5; ++i) {
        EXPECT_EQ(empty_res.ids[i], -1);
        EXPECT_TRUE(std::isinf(empty_res.dists[i]));
    }

    // Add 2 elements and query k = 5
    std::vector<float> v0(d, 1.0f);
    std::vector<float> v1(d, 2.0f);
    index.add(v0.data(), 10);
    index.add(v1.data(), 20);

    auto res = index.search(query.data(), 5);
    EXPECT_EQ(res.ids.size(), 5);
    EXPECT_EQ(res.ids[0], 10);
    EXPECT_EQ(res.ids[1], 20);
    EXPECT_EQ(res.ids[2], -1);
    EXPECT_EQ(res.ids[3], -1);
    EXPECT_EQ(res.ids[4], -1);
    EXPECT_TRUE(std::isinf(res.dists[2]));
    EXPECT_TRUE(std::isinf(res.dists[3]));
    EXPECT_TRUE(std::isinf(res.dists[4]));
}

TEST(FlatIndexTest, DeleteAndUpsert) {
    constexpr std::size_t d = 4;
    hnswdb::FlatIndex index(d, hnswdb::Metric::L2);

    std::vector<float> v0 = {0.0f, 0.0f, 0.0f, 0.0f};
    std::vector<float> v1 = {1.0f, 1.0f, 1.0f, 1.0f};
    std::vector<float> v2 = {2.0f, 2.0f, 2.0f, 2.0f};

    index.add(v0.data(), 0);
    index.add(v1.data(), 1);
    index.add(v2.data(), 2);
    EXPECT_EQ(index.size(), 3);
    EXPECT_TRUE(index.contains(1));

    // Delete label 1
    EXPECT_EQ(index.delete_labels({1}), 1);
    EXPECT_EQ(index.size(), 2);
    EXPECT_FALSE(index.contains(1));

    std::vector<float> query = {0.9f, 0.9f, 0.9f, 0.9f};
    auto res = index.search(query.data(), 2);
    // Deleted 1 must not appear
    EXPECT_NE(res.ids[0], 1);
    EXPECT_NE(res.ids[1], 1);

    // Upsert: re-add label 1 with new vector
    std::vector<float> v1_new = {0.9f, 0.9f, 0.9f, 0.9f};
    index.add(v1_new.data(), 1);
    EXPECT_EQ(index.size(), 3);
    EXPECT_TRUE(index.contains(1));

    res = index.search(query.data(), 1);
    EXPECT_EQ(res.ids[0], 1);
    EXPECT_NEAR(res.dists[0], 0.0f, 1e-5f);
}

TEST(FlatIndexTest, BatchSearchMultiThreaded) {
    constexpr std::size_t n = 500;
    constexpr std::size_t d = 32;
    constexpr std::size_t nq = 50;
    constexpr std::size_t k = 10;

    auto data = make_random_matrix(n, d, 77);
    auto queries = make_random_matrix(nq, d, 88);

    hnswdb::FlatIndex index(d, hnswdb::Metric::L2);
    for (std::size_t i = 0; i < n; ++i) {
        index.add(data.data() + i * d, static_cast<hnswdb::label_t>(i));
    }

    auto batch_res = index.search_batch(queries.data(), nq, k, 4);
    EXPECT_EQ(batch_res.size(), nq);

    for (std::size_t q = 0; q < nq; ++q) {
        auto single_res = index.search(queries.data() + q * d, k);
        EXPECT_EQ(batch_res[q].ids, single_res.ids);
        for (std::size_t i = 0; i < k; ++i) {
            EXPECT_NEAR(batch_res[q].dists[i], single_res.dists[i], 1e-5f);
        }
    }
}

TEST(FlatIndexTest, SaveAndLoad) {
    constexpr std::size_t n = 100;
    constexpr std::size_t d = 16;
    constexpr std::size_t k = 5;

    auto data = make_random_matrix(n, d, 999);
    auto queries = make_random_matrix(5, d, 888);

    std::string test_path = "test_flat_index_persist.hdb";

    {
        hnswdb::FlatIndex index(d, hnswdb::Metric::L2);
        for (std::size_t i = 0; i < n; ++i) {
            index.add(data.data() + i * d, static_cast<hnswdb::label_t>(i * 10));
        }
        index.delete_labels({20, 50});
        index.save(test_path);
    }

    auto loaded = hnswdb::FlatIndex::load(test_path);
    ASSERT_NE(loaded, nullptr);
    EXPECT_EQ(loaded->size(), n - 2);
    EXPECT_FALSE(loaded->contains(20));
    EXPECT_FALSE(loaded->contains(50));
    EXPECT_TRUE(loaded->contains(0));
    EXPECT_TRUE(loaded->contains(10));

    for (std::size_t q = 0; q < 5; ++q) {
        auto res = loaded->search(queries.data() + q * d, k);
        EXPECT_EQ(res.ids.size(), k);
        for (auto id : res.ids) {
            EXPECT_NE(id, 20);
            EXPECT_NE(id, 50);
        }
    }

    std::filesystem::remove(test_path);
}
