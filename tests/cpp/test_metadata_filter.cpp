#include <gtest/gtest.h>
#include "hnswdb/hnsw_index.h"
#include "hnswdb/metadata.h"
#include "hnswdb/types.h"
#include <random>
#include <vector>
#include <algorithm>
#include <filesystem>
#include <unordered_set>

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

TEST(MetadataFilterTest, AllOperatorsAndCombinators) {
    hnswdb::MetadataStore store;

    // Item 0: category="shoes", price=45.0, in_stock=true, rating=4
    hnswdb::MetaMap m0 = {
        {"category", std::string("shoes")},
        {"price", 45.0},
        {"in_stock", true},
        {"rating", int64_t(4)}
    };
    store.add_metadata(0, m0);

    // Item 1: category="shirts", price=25.0, in_stock=false, rating=5
    hnswdb::MetaMap m1 = {
        {"category", std::string("shirts")},
        {"price", 25.0},
        {"in_stock", false},
        {"rating", int64_t(5)}
    };
    store.add_metadata(1, m1);

    // Item 2: category="shoes", price=120.0, in_stock=true, rating=3
    hnswdb::MetaMap m2 = {
        {"category", std::string("shoes")},
        {"price", 120.0},
        {"in_stock", true},
        {"rating", int64_t(3)}
    };
    store.add_metadata(2, m2);

    // $eq
    auto eq_shoes = hnswdb::FilterExpr::Eq("category", std::string("shoes"));
    EXPECT_TRUE(store.evaluate(0, eq_shoes));
    EXPECT_FALSE(store.evaluate(1, eq_shoes));
    EXPECT_TRUE(store.evaluate(2, eq_shoes));

    // $ne
    auto ne_shoes = hnswdb::FilterExpr::Ne("category", std::string("shoes"));
    EXPECT_FALSE(store.evaluate(0, ne_shoes));
    EXPECT_TRUE(store.evaluate(1, ne_shoes));

    // $lt and $gte
    auto lt_50 = hnswdb::FilterExpr::Lt("price", 50.0);
    EXPECT_TRUE(store.evaluate(0, lt_50));
    EXPECT_TRUE(store.evaluate(1, lt_50));
    EXPECT_FALSE(store.evaluate(2, lt_50));

    auto gte_45 = hnswdb::FilterExpr::Gte("price", 45.0);
    EXPECT_TRUE(store.evaluate(0, gte_45));
    EXPECT_FALSE(store.evaluate(1, gte_45));
    EXPECT_TRUE(store.evaluate(2, gte_45));

    // $in
    auto in_cats = hnswdb::FilterExpr::In("category", {std::string("shoes"), std::string("hats")});
    EXPECT_TRUE(store.evaluate(0, in_cats));
    EXPECT_FALSE(store.evaluate(1, in_cats));
    EXPECT_TRUE(store.evaluate(2, in_cats));

    // $and: category == "shoes" AND price < 50
    auto and_expr = hnswdb::FilterExpr::And({eq_shoes, lt_50});
    EXPECT_TRUE(store.evaluate(0, and_expr));
    EXPECT_FALSE(store.evaluate(1, and_expr));
    EXPECT_FALSE(store.evaluate(2, and_expr));

    // $or: category == "shirts" OR price > 100
    auto gt_100 = hnswdb::FilterExpr::Gt("price", 100.0);
    auto or_expr = hnswdb::FilterExpr::Or({hnswdb::FilterExpr::Eq("category", std::string("shirts")), gt_100});
    EXPECT_FALSE(store.evaluate(0, or_expr));
    EXPECT_TRUE(store.evaluate(1, or_expr));
    EXPECT_TRUE(store.evaluate(2, or_expr));

    // $not: NOT (in_stock == true)
    auto not_in_stock = hnswdb::FilterExpr::Not(hnswdb::FilterExpr::Eq("in_stock", true));
    EXPECT_FALSE(store.evaluate(0, not_in_stock));
    EXPECT_TRUE(store.evaluate(1, not_in_stock));
    EXPECT_FALSE(store.evaluate(2, not_in_stock));
}

TEST(MetadataFilterTest, HighlySelectiveFilterExactResult) {
    constexpr std::size_t n = 1000;
    constexpr std::size_t d = 16;
    constexpr std::size_t k = 5;

    hnswdb::HNSWIndex idx(d, hnswdb::Metric::L2, n, 16, 200, 50, 42);
    auto data = make_random_matrix(n, d, 1);

    // Only 5 items have tag="vip" (0.5% selectivity)
    for (std::size_t i = 0; i < n; ++i) {
        hnswdb::MetaMap meta;
        if (i % 200 == 0) {
            meta["tag"] = std::string("vip");
        } else {
            meta["tag"] = std::string("regular");
        }
        idx.add(data.data() + i * d, static_cast<hnswdb::label_t>(i), &meta);
    }

    auto vip_filter = hnswdb::FilterExpr::Eq("tag", std::string("vip"));
    auto query = make_random_matrix(1, d, 99);

    auto res = idx.search(query.data(), k, 50, &vip_filter);

    // Verify all returned ids are VIP and exact matches
    for (hnswdb::label_t id : res.ids) {
        if (id != -1) {
            EXPECT_EQ(id % 200, 0);
            auto meta = idx.get_metadata(id);
            EXPECT_EQ(std::get<std::string>(meta["tag"]), "vip");
        }
    }
}

TEST(MetadataFilterTest, BroadFilterHighRecall) {
    constexpr std::size_t n = 2000;
    constexpr std::size_t d = 16;
    constexpr std::size_t k = 10;

    hnswdb::HNSWIndex idx(d, hnswdb::Metric::L2, n, 16, 200, 50, 42);
    auto data = make_random_matrix(n, d, 2);

    // 50% items are category A
    for (std::size_t i = 0; i < n; ++i) {
        hnswdb::MetaMap meta;
        meta["cat"] = (i % 2 == 0) ? std::string("A") : std::string("B");
        idx.add(data.data() + i * d, static_cast<hnswdb::label_t>(i), &meta);
    }

    auto filter_a = hnswdb::FilterExpr::Eq("cat", std::string("A"));
    auto queries = make_random_matrix(20, d, 88);

    for (std::size_t q = 0; q < 20; ++q) {
        auto res = idx.search(queries.data() + q * d, k, 100, &filter_a);
        for (hnswdb::label_t id : res.ids) {
            if (id != -1) {
                EXPECT_EQ(id % 2, 0);
            }
        }
    }
}

TEST(MetadataFilterTest, MetadataSurvivesLifecycle) {
    constexpr std::size_t n = 50;
    constexpr std::size_t d = 8;
    std::string test_file = "test_metadata_lifecycle.hdb";

    {
        hnswdb::HNSWIndex idx(d, hnswdb::Metric::L2, 100);
        auto data = make_random_matrix(n, d, 3);
        for (std::size_t i = 0; i < n; ++i) {
            hnswdb::MetaMap meta = {
                {"val", int64_t(i * 10)},
                {"label_str", "item_" + std::to_string(i)}
            };
            idx.add(data.data() + i * d, static_cast<hnswdb::label_t>(i), &meta);
        }

        // Upsert item 5
        hnswdb::MetaMap new_m5 = {{"val", int64_t(555)}, {"label_str", std::string("updated_5")}};
        idx.add(data.data() + 5 * d, 5, &new_m5);
        EXPECT_EQ(std::get<int64_t>(idx.get_metadata(5)["val"]), 555);

        // Delete item 10
        idx.delete_labels({10});
        EXPECT_FALSE(idx.has_metadata(10));

        idx.save(test_file);
    }

    auto loaded = hnswdb::HNSWIndex::load(test_file);
    ASSERT_NE(loaded, nullptr);
    EXPECT_TRUE(loaded->has_metadata(5));
    EXPECT_EQ(std::get<int64_t>(loaded->get_metadata(5)["val"]), 555);
    EXPECT_EQ(std::get<std::string>(loaded->get_metadata(5)["label_str"]), "updated_5");
    EXPECT_FALSE(loaded->has_metadata(10));

    // Test compact preserves metadata
    auto compacted = loaded->compact();
    ASSERT_NE(compacted, nullptr);
    EXPECT_TRUE(compacted->has_metadata(5));
    EXPECT_EQ(std::get<int64_t>(compacted->get_metadata(5)["val"]), 555);

    std::filesystem::remove(test_file);
}
