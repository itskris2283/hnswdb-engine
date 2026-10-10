#include <gtest/gtest.h>
#include "hnswdb/hnsw_index.h"
#include "hnswdb/flat_index.h"
#include "hnswdb/types.h"
#include <random>
#include <vector>
#include <cmath>
#include <unordered_set>
#include <iostream>

namespace {

std::vector<std::vector<float>> make_centers(std::size_t num_clusters, std::size_t d, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> norm_dist(0.0f, 1.0f);
    std::vector<std::vector<float>> centers(num_clusters, std::vector<float>(d));
    for (std::size_t c = 0; c < num_clusters; ++c) {
        for (std::size_t i = 0; i < d; ++i) {
            centers[c][i] = norm_dist(rng);
        }
    }
    return centers;
}

std::vector<float> sample_clustered(const std::vector<std::vector<float>>& centers, std::size_t n, std::size_t d, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> norm_dist(0.0f, 1.0f);
    std::uniform_int_distribution<std::size_t> cluster_dist(0, centers.size() - 1);

    std::vector<float> data(n * d);
    for (std::size_t i = 0; i < n; ++i) {
        std::size_t c = cluster_dist(rng);
        for (std::size_t dim = 0; dim < d; ++dim) {
            data[i * d + dim] = centers[c][dim] + 0.3f * norm_dist(rng);
        }
    }
    return data;
}

} // namespace

TEST(HnswInvariants, GraphValidationPasses) {
    constexpr std::size_t n = 1000;
    constexpr std::size_t d = 16;

    auto centers = make_centers(10, d, 42);
    auto data = sample_clustered(centers, n, d, 42);

    hnswdb::HNSWIndex idx(d, hnswdb::Metric::L2, n, /*M=*/16, /*efc=*/100, /*efs=*/50, /*seed=*/1);
    for (std::size_t i = 0; i < n; ++i) {
        idx.add(data.data() + i * d, static_cast<hnswdb::label_t>(i));
    }

    EXPECT_TRUE(idx.validate_graph());
    auto st = idx.stats();
    EXPECT_EQ(st["count"], n);
    EXPECT_GT(st["avg_degree_l0"], 0.0);
    EXPECT_LE(st["avg_degree_l0"], 32.0); // M0 = 32
}

TEST(HnswInvariants, GeometricLevelDistribution) {
    constexpr std::size_t n = 50000;
    constexpr std::size_t d = 4;
    constexpr std::size_t M = 16;

    hnswdb::HNSWIndex idx(d, hnswdb::Metric::L2, n, M, 100, 50, 42);

    double sum_levels = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        int lvl = idx.get_random_level();
        sum_levels += lvl;
    }

    double empirical_mean = sum_levels / n;
    // For geometric / exponential distribution with parameter mL = 1 / ln(M):
    // E[floor(-ln(U) * mL)] is approx 1 / (e^(1/mL) - 1) = 1 / (M - 1)
    double expected_mean = 1.0 / (static_cast<double>(M) - 1.0);

    double rel_diff = std::abs(empirical_mean - expected_mean) / expected_mean;
    EXPECT_LT(rel_diff, 0.20) << "Level distribution mean " << empirical_mean << " deviates >20% from expected " << expected_mean;
}

TEST(HnswInvariants, RecallGate20KSyntheticAtEf100) {
    constexpr std::size_t n = 20000;
    constexpr std::size_t d = 64;
    constexpr std::size_t nq = 500;
    constexpr std::size_t k = 10;
    constexpr std::size_t num_clusters = 50;

    auto centers = make_centers(num_clusters, d, 42);
    auto train_data = sample_clustered(centers, n, d, 100);
    auto query_data = sample_clustered(centers, nq, d, 200);

    hnswdb::HNSWIndex hnsw(d, hnswdb::Metric::L2, n, /*M=*/16, /*efc=*/200, /*efs=*/100, /*seed=*/1);
    hnswdb::FlatIndex flat(d, hnswdb::Metric::L2);

    for (std::size_t i = 0; i < n; ++i) {
        hnsw.add(train_data.data() + i * d, static_cast<hnswdb::label_t>(i));
        flat.add(train_data.data() + i * d, static_cast<hnswdb::label_t>(i));
    }

    EXPECT_TRUE(hnsw.validate_graph());

    double total_recall = 0.0;
    for (std::size_t q = 0; q < nq; ++q) {
        const float* q_vec = query_data.data() + q * d;
        auto hnsw_res = hnsw.search(q_vec, k, /*ef_search_override=*/100);
        auto flat_res = flat.search(q_vec, k);

        std::unordered_set<hnswdb::label_t> ground_truth(flat_res.ids.begin(), flat_res.ids.end());
        std::size_t hits = 0;
        for (hnswdb::label_t id : hnsw_res.ids) {
            if (ground_truth.count(id)) {
                ++hits;
            }
        }
        total_recall += static_cast<double>(hits) / k;
    }

    double mean_recall = total_recall / nq;
    std::cout << "[INFO] 20K Synthetic dataset Mean Recall@10 at ef=100: " << mean_recall << std::endl;
    EXPECT_GE(mean_recall, 0.95);
}

TEST(HnswInvariants, DuplicateAndSelfLinksZero) {
    constexpr std::size_t n = 500;
    constexpr std::size_t d = 16;
    auto centers = make_centers(5, d, 99);
    auto data = sample_clustered(centers, n, d, 99);

    hnswdb::HNSWIndex idx(d, hnswdb::Metric::L2, n, 16, 100, 50, 42);
    for (std::size_t i = 0; i < n; ++i) {
        idx.add(data.data() + i * d, static_cast<hnswdb::label_t>(i));
    }

    auto diag = idx.diagnostics();
    EXPECT_EQ(diag["duplicate_links_count"], 0.0);
    EXPECT_EQ(diag["self_links_count"], 0.0);
    EXPECT_TRUE(idx.validate_graph());
}

TEST(HnswInvariants, ParameterSafetyM) {
    constexpr std::size_t n = 100;
    constexpr std::size_t d = 8;
    auto data = sample_clustered(make_centers(2, d, 1), n, d, 1);

    for (std::size_t m_val : {8, 16, 64, 128}) {
        hnswdb::HNSWIndex idx(d, hnswdb::Metric::L2, n, m_val, /*efc=*/50, /*efs=*/20, /*seed=*/42);
        for (std::size_t i = 0; i < n; ++i) {
            idx.add(data.data() + i * d, static_cast<hnswdb::label_t>(i));
        }
        EXPECT_EQ(idx.size(), n);
        EXPECT_TRUE(idx.validate_graph());
        auto res = idx.search(data.data(), 5);
        EXPECT_EQ(res.ids.size(), 5);
        EXPECT_EQ(res.ids[0], 0);
    }
}

TEST(HnswInvariants, HandBuiltGraphReachability) {
    // Tests reachability logic: on a well-connected graph of 50 nodes, unreachable count is 0
    constexpr std::size_t n = 50;
    constexpr std::size_t d = 4;
    auto data = sample_clustered(make_centers(2, d, 42), n, d, 42);

    hnswdb::HNSWIndex idx(d, hnswdb::Metric::L2, n, 16, 100, 50, 42);
    for (std::size_t i = 0; i < n; ++i) {
        idx.add(data.data() + i * d, static_cast<hnswdb::label_t>(i));
    }

    auto diag = idx.diagnostics();
    EXPECT_EQ(diag["unreachable_nodes_count"], 0.0);
    EXPECT_EQ(diag["weakly_connected_components"], 1.0);
}

