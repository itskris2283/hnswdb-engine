#include "hnswdb/distance.h"
#include "hnswdb/types.h"
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <atomic>

#if defined(_MSC_VER)
#include <intrin.h>
#include <immintrin.h>
#elif defined(__GNUC__) || defined(__clang__)
#include <cpuid.h>
#include <immintrin.h>
#endif

namespace hnswdb {
namespace distance {

static std::atomic<bool> g_force_scalar{false};

void set_force_scalar(bool force) {
    g_force_scalar.store(force);
}

float l2_sqr_scalar(const float* a, const float* b, std::size_t dim) {
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;

    std::size_t i = 0;
    std::size_t unroll_end = dim >= 4 ? dim - 3 : 0;
    for (; i < unroll_end; i += 4) {
        float d0 = a[i] - b[i];
        float d1 = a[i + 1] - b[i + 1];
        float d2 = a[i + 2] - b[i + 2];
        float d3 = a[i + 3] - b[i + 3];
        sum0 += d0 * d0;
        sum1 += d1 * d1;
        sum2 += d2 * d2;
        sum3 += d3 * d3;
    }
    float sum = (sum0 + sum1) + (sum2 + sum3);
    for (; i < dim; ++i) {
        float d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
}

float dot_product_scalar(const float* a, const float* b, std::size_t dim) {
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;

    std::size_t i = 0;
    std::size_t unroll_end = dim >= 4 ? dim - 3 : 0;
    for (; i < unroll_end; i += 4) {
        sum0 += a[i] * b[i];
        sum1 += a[i + 1] * b[i + 1];
        sum2 += a[i + 2] * b[i + 2];
        sum3 += a[i + 3] * b[i + 3];
    }
    float sum = (sum0 + sum1) + (sum2 + sum3);
    for (; i < dim; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

#if defined(__x86_64__) || defined(_M_X64) || defined(HNSWDB_HAS_AVX2)

#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx2,fma")))
#endif
float l2_sqr_avx2(const float* a, const float* b, std::size_t dim) {
    std::size_t i = 0;
    __m256 sum256_0 = _mm256_setzero_ps();
    __m256 sum256_1 = _mm256_setzero_ps();

    std::size_t unroll16 = dim >= 16 ? dim - 15 : 0;
    for (; i < unroll16; i += 16) {
        __m256 va0 = _mm256_loadu_ps(a + i);
        __m256 vb0 = _mm256_loadu_ps(b + i);
        __m256 diff0 = _mm256_sub_ps(va0, vb0);
        sum256_0 = _mm256_fmadd_ps(diff0, diff0, sum256_0);

        __m256 va1 = _mm256_loadu_ps(a + i + 8);
        __m256 vb1 = _mm256_loadu_ps(b + i + 8);
        __m256 diff1 = _mm256_sub_ps(va1, vb1);
        sum256_1 = _mm256_fmadd_ps(diff1, diff1, sum256_1);
    }

    sum256_0 = _mm256_add_ps(sum256_0, sum256_1);

    std::size_t unroll8 = dim >= 8 ? dim - 7 : 0;
    for (; i < unroll8; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 diff = _mm256_sub_ps(va, vb);
        sum256_0 = _mm256_fmadd_ps(diff, diff, sum256_0);
    }

    // Horizontal reduction of sum256_0
    __m128 vlow = _mm256_castps256_ps128(sum256_0);
    __m128 vhigh = _mm256_extractf128_ps(sum256_0, 1);
    __m128 vsum = _mm_add_ps(vlow, vhigh);
    __m128 shuf = _mm_movehdup_ps(vsum);
    __m128 sums = _mm_add_ps(vsum, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    float total = _mm_cvtss_f32(sums);

    // Scalar tail loop
    for (; i < dim; ++i) {
        float d = a[i] - b[i];
        total += d * d;
    }

    return total;
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx2,fma")))
#endif
float dot_product_avx2(const float* a, const float* b, std::size_t dim) {
    std::size_t i = 0;
    __m256 sum256_0 = _mm256_setzero_ps();
    __m256 sum256_1 = _mm256_setzero_ps();

    std::size_t unroll16 = dim >= 16 ? dim - 15 : 0;
    for (; i < unroll16; i += 16) {
        __m256 va0 = _mm256_loadu_ps(a + i);
        __m256 vb0 = _mm256_loadu_ps(b + i);
        sum256_0 = _mm256_fmadd_ps(va0, vb0, sum256_0);

        __m256 va1 = _mm256_loadu_ps(a + i + 8);
        __m256 vb1 = _mm256_loadu_ps(b + i + 8);
        sum256_1 = _mm256_fmadd_ps(va1, vb1, sum256_1);
    }

    sum256_0 = _mm256_add_ps(sum256_0, sum256_1);

    std::size_t unroll8 = dim >= 8 ? dim - 7 : 0;
    for (; i < unroll8; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        sum256_0 = _mm256_fmadd_ps(va, vb, sum256_0);
    }

    // Horizontal reduction
    __m128 vlow = _mm256_castps256_ps128(sum256_0);
    __m128 vhigh = _mm256_extractf128_ps(sum256_0, 1);
    __m128 vsum = _mm_add_ps(vlow, vhigh);
    __m128 shuf = _mm_movehdup_ps(vsum);
    __m128 sums = _mm_add_ps(vsum, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    float total = _mm_cvtss_f32(sums);

    // Scalar tail loop
    for (; i < dim; ++i) {
        total += a[i] * b[i];
    }

    return total;
}

#endif // x86_64

bool is_avx2_supported() {
#if defined(_MSC_VER)
    int cpuInfo[4] = {0};
    __cpuid(cpuInfo, 0);
    int nIds = cpuInfo[0];
    if (nIds < 7) return false;

    __cpuid(cpuInfo, 1);
    bool osUsesXSAVE_XRSTORE = (cpuInfo[2] & (1 << 27)) || false;
    bool cpuAVX = (cpuInfo[2] & (1 << 28)) || false;
    bool cpuFMA = (cpuInfo[2] & (1 << 12)) || false;

    if (!osUsesXSAVE_XRSTORE || !cpuAVX || !cpuFMA) return false;

    unsigned long long xcrFeatureMask = _xgetbv(0);
    bool avxSupported = (xcrFeatureMask & 6) == 6;
    if (!avxSupported) return false;

    __cpuidex(cpuInfo, 7, 0);
    bool cpuAVX2 = (cpuInfo[1] & (1 << 5)) || false;
    return cpuAVX2;

#elif defined(__GNUC__) || defined(__clang__)
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#else
    return false;
#endif
}

static bool check_force_scalar_env() {
    if (g_force_scalar.load()) return true;
    const char* env = std::getenv("HNSWDB_FORCE_SCALAR");
    if (env && (std::strcmp(env, "1") == 0 || std::strcmp(env, "true") == 0 || std::strcmp(env, "TRUE") == 0)) {
        return true;
    }
    return false;
}

DistFunc get_l2sq_func() {
    if (check_force_scalar_env()) {
        return l2_sqr_scalar;
    }
#if defined(__x86_64__) || defined(_M_X64) || defined(HNSWDB_HAS_AVX2)
    if (is_avx2_supported()) {
        return l2_sqr_avx2;
    }
#endif
    return l2_sqr_scalar;
}

DistFunc get_dot_func() {
    if (check_force_scalar_env()) {
        return dot_product_scalar;
    }
#if defined(__x86_64__) || defined(_M_X64) || defined(HNSWDB_HAS_AVX2)
    if (is_avx2_supported()) {
        return dot_product_avx2;
    }
#endif
    return dot_product_scalar;
}

DistFunc get_ip_func() {
    return get_dot_func();
}

const char* active_kernel_name() {
    if (check_force_scalar_env()) {
        return "scalar";
    }
#if defined(__x86_64__) || defined(_M_X64) || defined(HNSWDB_HAS_AVX2)
    if (is_avx2_supported()) {
        return "avx2";
    }
#endif
    return "scalar";
}

void normalize_vector(const float* src, float* dst, std::size_t dim) {
    double norm_sq = 0.0;
    for (std::size_t i = 0; i < dim; ++i) {
        norm_sq += static_cast<double>(src[i]) * static_cast<double>(src[i]);
    }
    if (norm_sq <= 1e-24 || std::isnan(norm_sq) || std::isinf(norm_sq)) {
        throw InvalidVector("Zero or invalid vector cannot be normalized for cosine metric");
    }
    float inv_norm = static_cast<float>(1.0 / std::sqrt(norm_sq));
    for (std::size_t i = 0; i < dim; ++i) {
        dst[i] = src[i] * inv_norm;
    }
}

bool validate_vector(const float* vec, std::size_t dim) {
    if (!vec) return false;
    for (std::size_t i = 0; i < dim; ++i) {
        if (std::isnan(vec[i]) || std::isinf(vec[i])) {
            return false;
        }
    }
    return true;
}

void prefetch_vector(const float* vec, std::size_t dim) {
#if defined(_MSC_VER) || defined(__GNUC__) || defined(__clang__)
    const char* ptr = reinterpret_cast<const char*>(vec);
    std::size_t bytes = dim * sizeof(float);
    for (std::size_t i = 0; i < bytes; i += 64) {
        #if defined(_MSC_VER)
        _mm_prefetch(ptr + i, _MM_HINT_T0);
        #elif defined(__GNUC__) || defined(__clang__)
        __builtin_prefetch(ptr + i, 0, 3);
        #endif
    }
#else
    (void)vec;
    (void)dim;
#endif
}

} // namespace distance
} // namespace hnswdb
