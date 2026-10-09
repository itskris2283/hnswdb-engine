#pragma once
#include <cstddef>
#include <cstdint>
#include <string>

namespace hnswdb {
namespace distance {

using DistFunc = float (*)(const float*, const float*, std::size_t);

// Core distance computation functions
float l2_sqr_scalar(const float* a, const float* b, std::size_t dim);
float dot_product_scalar(const float* a, const float* b, std::size_t dim);

#if defined(__x86_64__) || defined(_M_X64) || defined(HNSWDB_HAS_AVX2)
float l2_sqr_avx2(const float* a, const float* b, std::size_t dim);
float dot_product_avx2(const float* a, const float* b, std::size_t dim);
#endif

// Runtime dispatch
DistFunc get_l2sq_func();
DistFunc get_dot_func();
DistFunc get_ip_func();

const char* active_kernel_name();
void set_force_scalar(bool force);
bool is_avx2_supported();

// Vector normalization and validation helpers
void normalize_vector(const float* src, float* dst, std::size_t dim);
bool validate_vector(const float* vec, std::size_t dim);

// Prefetch helper
void prefetch_vector(const float* vec, std::size_t dim);

} // namespace distance
} // namespace hnswdb
