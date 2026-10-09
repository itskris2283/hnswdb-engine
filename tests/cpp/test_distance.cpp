#include <gtest/gtest.h>
#include "hnswdb/distance.h"
#include "hnswdb/types.h"
#include <random>
#include <vector>
#include <cmath>

namespace {

double double_l2_sqr(const float* a, const float* b, std::size_t dim) {
    double sum = 0.0;
    for (std::size_t i = 0; i < dim; ++i) {
        double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
        sum += d * d;
    }
    return sum;
}

double double_dot_product(const float* a, const float* b, std::size_t dim) {
    double sum = 0.0;
    for (std::size_t i = 0; i < dim; ++i) {
        sum += static_cast<double>(a[i]) * static_cast<double>(b[i]);
    }
    return sum;
}

std::vector<float> generate_random_vector(std::size_t dim, float min_val, float max_val, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> dist(min_val, max_val);
    std::vector<float> v(dim);
    for (std::size_t i = 0; i < dim; ++i) {
        v[i] = dist(rng);
    }
    return v;
}

} // namespace

TEST(DistanceTest, ScalarVsDoubleReference) {
    const std::vector<std::size_t> test_dims = {1, 2, 7, 8, 9, 15, 16, 17, 31, 100, 128, 960};

    for (std::size_t dim : test_dims) {
        auto a = generate_random_vector(dim, -5.0f, 5.0f, 42 + dim);
        auto b = generate_random_vector(dim, -5.0f, 5.0f, 1337 + dim);

        float l2_scalar = hnswdb::distance::l2_sqr_scalar(a.data(), b.data(), dim);
        double l2_ref = double_l2_sqr(a.data(), b.data(), dim);
        double l2_rel_err = std::abs(l2_scalar - l2_ref) / std::max(1.0, l2_ref);
        EXPECT_LT(l2_rel_err, 1e-4) << "L2 scalar relative error failed for dim " << dim;

        float dot_scalar = hnswdb::distance::dot_product_scalar(a.data(), b.data(), dim);
        double dot_ref = double_dot_product(a.data(), b.data(), dim);
        double dot_rel_err = std::abs(dot_scalar - dot_ref) / std::max(1.0, std::abs(dot_ref));
        EXPECT_LT(dot_rel_err, 1e-4) << "Dot scalar relative error failed for dim " << dim;
    }
}

#if defined(__x86_64__) || defined(_M_X64) || defined(HNSWDB_HAS_AVX2)
TEST(DistanceTest, AVX2VsScalarVsDouble) {
    if (!hnswdb::distance::is_avx2_supported()) {
        GTEST_SKIP() << "AVX2 not supported on this CPU";
    }

    const std::vector<std::size_t> test_dims = {1, 2, 7, 8, 9, 15, 16, 17, 31, 100, 128, 960};

    for (std::size_t dim : test_dims) {
        auto a = generate_random_vector(dim, -10.0f, 10.0f, 101 + dim);
        auto b = generate_random_vector(dim, -10.0f, 10.0f, 202 + dim);

        float l2_scalar = hnswdb::distance::l2_sqr_scalar(a.data(), b.data(), dim);
        float l2_avx2 = hnswdb::distance::l2_sqr_avx2(a.data(), b.data(), dim);
        double l2_ref = double_l2_sqr(a.data(), b.data(), dim);

        double l2_rel_err = std::abs(l2_avx2 - l2_scalar) / std::max(1.0f, l2_scalar);
        EXPECT_LT(l2_rel_err, 1e-4) << "AVX2 vs Scalar L2 mismatch at dim " << dim;

        double l2_ref_err = std::abs(l2_avx2 - l2_ref) / std::max(1.0, l2_ref);
        EXPECT_LT(l2_ref_err, 1e-4) << "AVX2 vs Double L2 mismatch at dim " << dim;

        float dot_scalar = hnswdb::distance::dot_product_scalar(a.data(), b.data(), dim);
        float dot_avx2 = hnswdb::distance::dot_product_avx2(a.data(), b.data(), dim);
        double dot_ref = double_dot_product(a.data(), b.data(), dim);

        double dot_rel_err = std::abs(dot_avx2 - dot_scalar) / std::max(1.0f, std::abs(dot_scalar));
        EXPECT_LT(dot_rel_err, 1e-4) << "AVX2 vs Scalar Dot mismatch at dim " << dim;

        double dot_ref_err = std::abs(dot_avx2 - dot_ref) / std::max(1.0, std::abs(dot_ref));
        EXPECT_LT(dot_ref_err, 1e-4) << "AVX2 vs Double Dot mismatch at dim " << dim;
    }
}
#endif

TEST(DistanceTest, EdgeCasesZerosAndLargeMagnitudes) {
    const std::vector<std::size_t> test_dims = {1, 8, 16, 128};

    for (std::size_t dim : test_dims) {
        // All zeros
        std::vector<float> zeros_a(dim, 0.0f);
        std::vector<float> zeros_b(dim, 0.0f);
        EXPECT_FLOAT_EQ(hnswdb::distance::l2_sqr_scalar(zeros_a.data(), zeros_b.data(), dim), 0.0f);
        EXPECT_FLOAT_EQ(hnswdb::distance::dot_product_scalar(zeros_a.data(), zeros_b.data(), dim), 0.0f);

        // Large values (e.g., 1000.0)
        std::vector<float> large_a(dim, 1000.0f);
        std::vector<float> large_b(dim, 1000.0f);
        EXPECT_FLOAT_EQ(hnswdb::distance::l2_sqr_scalar(large_a.data(), large_b.data(), dim), 0.0f);
        EXPECT_FLOAT_EQ(hnswdb::distance::dot_product_scalar(large_a.data(), large_b.data(), dim), static_cast<float>(dim) * 1000000.0f);
    }
}

TEST(DistanceTest, RuntimeDispatchAndForceScalar) {
    hnswdb::distance::set_force_scalar(false);
    std::string default_kernel = hnswdb::distance::active_kernel_name();

    hnswdb::distance::set_force_scalar(true);
    EXPECT_STREQ(hnswdb::distance::active_kernel_name(), "scalar");

    hnswdb::distance::set_force_scalar(false);
    EXPECT_EQ(default_kernel, hnswdb::distance::active_kernel_name());
}

TEST(DistanceTest, NormalizationAndValidation) {
    std::vector<float> v = {3.0f, 4.0f};
    std::vector<float> norm_v(2);
    hnswdb::distance::normalize_vector(v.data(), norm_v.data(), 2);
    EXPECT_NEAR(norm_v[0], 0.6f, 1e-5f);
    EXPECT_NEAR(norm_v[1], 0.8f, 1e-5f);

    // Zero vector normalization should throw InvalidVector
    std::vector<float> zero_v = {0.0f, 0.0f};
    EXPECT_THROW(hnswdb::distance::normalize_vector(zero_v.data(), norm_v.data(), 2), hnswdb::InvalidVector);

    // NaN / Inf validation
    std::vector<float> nan_v = {1.0f, std::numeric_limits<float>::quiet_NaN()};
    EXPECT_FALSE(hnswdb::distance::validate_vector(nan_v.data(), 2));

    std::vector<float> inf_v = {1.0f, std::numeric_limits<float>::infinity()};
    EXPECT_FALSE(hnswdb::distance::validate_vector(inf_v.data(), 2));
}
