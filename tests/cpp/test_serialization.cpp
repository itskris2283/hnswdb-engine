#include <gtest/gtest.h>
#include "hnswdb/hnsw_index.h"
#include "hnswdb/flat_index.h"
#include "hnswdb/serialization.h"
#include "hnswdb/types.h"
#include <random>
#include <vector>
#include <fstream>
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

TEST(SerializationTest, BitIdenticalRoundTrip) {
    constexpr std::size_t n = 200;
    constexpr std::size_t d = 16;
    constexpr std::size_t nq = 20;
    constexpr std::size_t k = 10;

    auto data = make_random_matrix(n, d, 42);
    auto queries = make_random_matrix(nq, d, 99);

    std::string test_file = "test_hnsw_roundtrip.hdb";

    std::vector<hnswdb::SearchResult> before_results(nq);
    {
        hnswdb::HNSWIndex idx(d, hnswdb::Metric::L2, n, 16, 200, 50, 42);
        for (std::size_t i = 0; i < n; ++i) {
            idx.add(data.data() + i * d, static_cast<hnswdb::label_t>(i * 3));
        }
        // Delete a few items
        idx.delete_labels({3, 9, 30});

        for (std::size_t q = 0; q < nq; ++q) {
            before_results[q] = idx.search(queries.data() + q * d, k);
        }

        idx.save(test_file);
        EXPECT_FALSE(std::filesystem::exists(test_file + ".tmp"));
    }

    auto loaded = hnswdb::HNSWIndex::load(test_file);
    ASSERT_NE(loaded, nullptr);
    EXPECT_EQ(loaded->size(), n - 3);

    for (std::size_t q = 0; q < nq; ++q) {
        auto after_res = loaded->search(queries.data() + q * d, k);
        EXPECT_EQ(before_results[q].ids, after_res.ids);
        for (std::size_t i = 0; i < k; ++i) {
            EXPECT_EQ(before_results[q].dists[i], after_res.dists[i]); // Bit-for-bit identical
        }
    }

    std::filesystem::remove(test_file);
}

TEST(SerializationTest, TruncatedFile) {
    constexpr std::size_t n = 50;
    constexpr std::size_t d = 8;
    std::string test_file = "test_truncated.hdb";

    {
        hnswdb::HNSWIndex idx(d, hnswdb::Metric::L2, n);
        auto data = make_random_matrix(n, d, 1);
        for (std::size_t i = 0; i < n; ++i) {
            idx.add(data.data() + i * d, static_cast<hnswdb::label_t>(i));
        }
        idx.save(test_file);
    }

    // Truncate file by half
    std::uintmax_t orig_size = std::filesystem::file_size(test_file);
    std::filesystem::resize_file(test_file, orig_size / 2);

    EXPECT_THROW(hnswdb::HNSWIndex::load(test_file), hnswdb::CorruptFile);
    std::filesystem::remove(test_file);
}

TEST(SerializationTest, FlippedSingleByteCrcCatches) {
    constexpr std::size_t n = 50;
    constexpr std::size_t d = 8;
    std::string test_file = "test_corrupt_byte.hdb";

    {
        hnswdb::HNSWIndex idx(d, hnswdb::Metric::L2, n);
        auto data = make_random_matrix(n, d, 2);
        for (std::size_t i = 0; i < n; ++i) {
            idx.add(data.data() + i * d, static_cast<hnswdb::label_t>(i));
        }
        idx.save(test_file);
    }

    // Read entire file and flip a single byte in the middle
    std::vector<uint8_t> buffer;
    {
        std::ifstream ifs(test_file, std::ios::binary);
        buffer.assign((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    }
    ASSERT_GT(buffer.size(), 40u);
    buffer[35] ^= 0xFF; // Flip byte in payload

    {
        std::ofstream ofs(test_file, std::ios::binary | std::ios::trunc);
        ofs.write(reinterpret_cast<const char*>(buffer.data()), buffer.size());
    }

    EXPECT_THROW(hnswdb::HNSWIndex::load(test_file), hnswdb::CorruptFile);
    std::filesystem::remove(test_file);
}

TEST(SerializationTest, WrongMagicHeader) {
    std::string test_file = "test_wrong_magic.hdb";
    std::vector<uint8_t> bad_data(128, 0);
    std::memcpy(bad_data.data(), "BADMAGIC", 8);

    {
        std::ofstream ofs(test_file, std::ios::binary | std::ios::trunc);
        ofs.write(reinterpret_cast<const char*>(bad_data.data()), bad_data.size());
    }

    EXPECT_THROW(hnswdb::HNSWIndex::load(test_file), hnswdb::CorruptFile);
    std::filesystem::remove(test_file);
}

TEST(SerializationTest, FutureVersionUnsupported) {
    constexpr std::size_t n = 20;
    constexpr std::size_t d = 4;
    std::string test_file = "test_future_version.hdb";

    {
        hnswdb::HNSWIndex idx(d, hnswdb::Metric::L2, n);
        auto data = make_random_matrix(n, d, 3);
        for (std::size_t i = 0; i < n; ++i) {
            idx.add(data.data() + i * d, static_cast<hnswdb::label_t>(i));
        }
        idx.save(test_file);
    }

    // Modify version field (offset 8) to 999 and recompute CRC
    std::vector<uint8_t> buffer;
    {
        std::ifstream ifs(test_file, std::ios::binary);
        buffer.assign((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    }
    uint32_t future_version = 999;
    std::memcpy(buffer.data() + 8, &future_version, sizeof(future_version));

    // Update CRC
    uint32_t new_crc = hnswdb::CRC32::compute(buffer.data(), buffer.size() - 4);
    std::memcpy(buffer.data() + buffer.size() - 4, &new_crc, sizeof(new_crc));

    {
        std::ofstream ofs(test_file, std::ios::binary | std::ios::trunc);
        ofs.write(reinterpret_cast<const char*>(buffer.data()), buffer.size());
    }

    EXPECT_THROW(hnswdb::HNSWIndex::load(test_file), hnswdb::UnsupportedVersion);
    std::filesystem::remove(test_file);
}

TEST(SerializationTest, ZeroLengthFile) {
    std::string test_file = "test_zero_length.hdb";
    {
        std::ofstream ofs(test_file, std::ios::binary | std::ios::trunc);
    }

    EXPECT_THROW(hnswdb::HNSWIndex::load(test_file), hnswdb::CorruptFile);
    std::filesystem::remove(test_file);
}

TEST(SerializationTest, BogusHugeLengthField) {
    constexpr std::size_t n = 10;
    constexpr std::size_t d = 4;
    std::string test_file = "test_huge_length.hdb";

    {
        hnswdb::HNSWIndex idx(d, hnswdb::Metric::L2, n);
        auto data = make_random_matrix(n, d, 4);
        for (std::size_t i = 0; i < n; ++i) {
            idx.add(data.data() + i * d, static_cast<hnswdb::label_t>(i));
        }
        idx.save(test_file);
    }

    std::vector<uint8_t> buffer;
    {
        std::ifstream ifs(test_file, std::ios::binary);
        buffer.assign((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    }

    // Set first section size (offset 60) to 0xFFFFFFFFFFFFFFFF
    uint64_t huge_len = 0xFFFFFFFFFFFFFFFFULL;
    std::memcpy(buffer.data() + 60, &huge_len, sizeof(huge_len));

    uint32_t new_crc = hnswdb::CRC32::compute(buffer.data(), buffer.size() - 4);
    std::memcpy(buffer.data() + buffer.size() - 4, &new_crc, sizeof(new_crc));

    {
        std::ofstream ofs(test_file, std::ios::binary | std::ios::trunc);
        ofs.write(reinterpret_cast<const char*>(buffer.data()), buffer.size());
    }

    EXPECT_THROW(hnswdb::HNSWIndex::load(test_file), hnswdb::CorruptFile);
    std::filesystem::remove(test_file);
}
