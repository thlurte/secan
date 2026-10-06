#pragma once

#include <cstddef>
#include <cstdint>

namespace secan {

/**
 * @brief Scalar reference for L2 squared integer distance between two 8-bit quantized vectors.
 * sum_{i=0}^{dim-1} (a[i] - b[i])^2
 */
uint32_t l2_squared_sq8_scalar(const uint8_t* a, const uint8_t* b, size_t dim) noexcept;

/**
 * @brief AVX2 vectorized L2 squared integer distance processing 32 dimensions per iteration.
 * Uses _mm256_subs_epu8 / _mm256_madd_epi16 with widening to 32-bit accumulators to prevent overflow.
 */
uint32_t l2_squared_sq8_avx2(const uint8_t* a, const uint8_t* b, size_t dim) noexcept;

/**
 * @brief Default L2 squared distance dispatch for SQ8 vectors.
 */
inline uint32_t l2_squared_sq8(const uint8_t* a, const uint8_t* b, size_t dim) noexcept {
    return l2_squared_sq8_avx2(a, b, dim);
}

/**
 * @brief Scalar reference for Inner Product integer distance between two 8-bit quantized vectors.
 * sum_{i=0}^{dim-1} a[i] * b[i]
 */
uint32_t inner_product_sq8_scalar(const uint8_t* a, const uint8_t* b, size_t dim) noexcept;

/**
 * @brief AVX2 vectorized Inner Product integer distance for SQ8 vectors.
 */
uint32_t inner_product_sq8_avx2(const uint8_t* a, const uint8_t* b, size_t dim) noexcept;

/**
 * @brief Default Inner Product distance dispatch for SQ8 vectors.
 */
inline uint32_t inner_product_sq8(const uint8_t* a, const uint8_t* b, size_t dim) noexcept {
    return inner_product_sq8_avx2(a, b, dim);
}

} // namespace secan
