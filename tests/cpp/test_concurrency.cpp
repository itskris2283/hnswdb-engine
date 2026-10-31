#include <gtest/gtest.h>
#include "hnswdb/hnsw_index.h"
#include "hnswdb/flat_index.h"
#include "hnswdb/thread_pool.h"
#include "hnswdb/types.h"
#include <random>
#include <vector>
#include <thread>
#include <atomic>
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

TEST(ConcurrencyTest, ThreadPoolBasic) {
    hnswdb::ThreadPool pool(4);
    std::vector<std::future<int>> results;
    for (int i = 0; i < 16; ++i) {
        results.push_back(pool.submit([i]() {
            return i * i;
        }));
    }
    for (int i = 0; i < 16; ++i) {
        EXPECT_EQ(results[i].get(), i * i);
    }
}

TEST(ConcurrencyTest, ConcurrentDisjointWritersAndReaders) {
    static constexpr std::size_t d = 16;
    static constexpr std::size_t num_writers = 8;
    static constexpr std::size_t items_per_writer = 250;
    static constexpr std::size_t total_items = num_writers * items_per_writer;
    static constexpr std::size_t num_readers = 8;

    auto data = make_random_matrix(total_items, d, 42);

    hnswdb::HNSWIndex idx(d, hnswdb::Metric::L2, total_items, 16, 200, 50, 42);
    hnswdb::FlatIndex flat(d, hnswdb::Metric::L2);

    // Populate a base of 100 items first
    for (std::size_t i = 0; i < 100; ++i) {
        idx.add(data.data() + i * d, static_cast<hnswdb::label_t>(i));
        flat.add(data.data() + i * d, static_cast<hnswdb::label_t>(i));
    }

    std::atomic<bool> stop_readers{false};
    std::atomic<std::size_t> search_count{0};

    // Start 8 reader threads
    std::vector<std::thread> readers;
    for (std::size_t r = 0; r < num_readers; ++r) {
        readers.emplace_back([&idx, &data, &stop_readers, &search_count, r]() {
            std::size_t query_idx = (r * 10) % 100;
            while (!stop_readers.load(std::memory_order_relaxed)) {
                auto res = idx.search(data.data() + query_idx * d, 5);
                EXPECT_EQ(res.ids.size(), 5);
                search_count.fetch_add(1, std::memory_order_relaxed);
                std::this_thread::yield();
            }
        });
    }

    // Start 8 writer threads inserting disjoint ranges
    std::vector<std::thread> writers;
    for (std::size_t w = 0; w < num_writers; ++w) {
        writers.emplace_back([&idx, &data, w]() {
            std::size_t start = 100 + w * ((total_items - 100) / num_writers);
            std::size_t end = (w == num_writers - 1) ? total_items : start + ((total_items - 100) / num_writers);
            for (std::size_t i = start; i < end; ++i) {
                idx.add(data.data() + i * d, static_cast<hnswdb::label_t>(i));
            }
        });
    }

    for (auto& w : writers) {
        w.join();
    }

    stop_readers.store(true);
    for (auto& r : readers) {
        r.join();
    }

    EXPECT_GT(search_count.load(), 100u);
    EXPECT_EQ(idx.size(), total_items);
    EXPECT_TRUE(idx.validate_graph());
}

TEST(ConcurrencyTest, ConcurrentSearchAndDelete) {
    static constexpr std::size_t d = 8;
    static constexpr std::size_t n = 500;

    auto data = make_random_matrix(n, d, 101);
    hnswdb::HNSWIndex idx(d, hnswdb::Metric::L2, n, 16, 200, 50, 42);

    for (std::size_t i = 0; i < n; ++i) {
        idx.add(data.data() + i * d, static_cast<hnswdb::label_t>(i));
    }

    std::atomic<bool> done{false};
    std::vector<std::thread> readers;
    for (int r = 0; r < 4; ++r) {
        readers.emplace_back([&idx, &data, &done, r]() {
            while (!done.load(std::memory_order_relaxed)) {
                auto res = idx.search(data.data() + (r * 10) * d, 5);
                EXPECT_EQ(res.ids.size(), 5);
                for (hnswdb::label_t id : res.ids) {
                    if (id != -1) {
                        EXPECT_GE(id, 0);
                        EXPECT_LT(id, static_cast<hnswdb::label_t>(n));
                    }
                }
                std::this_thread::yield();
            }
        });
    }

    // Delete half the elements concurrently
    std::thread deleter([&idx]() {
        for (std::size_t i = 0; i < n / 2; ++i) {
            idx.delete_labels({static_cast<hnswdb::label_t>(i * 2)});
            std::this_thread::yield();
        }
    });

    deleter.join();
    done.store(true);
    for (auto& r : readers) {
        r.join();
    }

    EXPECT_EQ(idx.size(), n - (n / 2));
}

TEST(ConcurrencyTest, ConcurrentSearchDuringSave) {
    static constexpr std::size_t d = 8;
    static constexpr std::size_t n = 300;

    auto data = make_random_matrix(n, d, 202);
    hnswdb::HNSWIndex idx(d, hnswdb::Metric::L2, n, 16, 200, 50, 42);

    for (std::size_t i = 0; i < n; ++i) {
        idx.add(data.data() + i * d, static_cast<hnswdb::label_t>(i));
    }

    std::string filename = "test_concurrent_save.hdb";
    std::atomic<bool> done{false};

    std::vector<std::thread> readers;
    for (int r = 0; r < 4; ++r) {
        readers.emplace_back([&idx, &data, &done, r]() {
            while (!done.load(std::memory_order_relaxed)) {
                auto res = idx.search(data.data() + (r * 10) * d, 5);
                EXPECT_EQ(res.ids.size(), 5);
                std::this_thread::yield();
            }
        });
    }

    std::thread saver([&idx, &filename]() {
        idx.save(filename);
    });

    saver.join();
    done.store(true);
    for (auto& r : readers) {
        r.join();
    }

    EXPECT_TRUE(std::filesystem::exists(filename));
    auto loaded = hnswdb::HNSWIndex::load(filename);
    ASSERT_NE(loaded, nullptr);
    EXPECT_EQ(loaded->size(), n);

    std::filesystem::remove(filename);
}
