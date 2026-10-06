#include "secan/quantization/distance_sq8.h"
#include <immintrin.h>

namespace secan {

uint32_t l2_squared_sq8_scalar(const uint8_t* a, const uint8_t* b, size_t dim) noexcept {
    if (a == nullptr || b == nullptr || dim == 0) return 0;
    uint32_t total = 0;
    for (size_t i = 0; i < dim; ++i) {
        int32_t diff = static_cast<int32_t>(a[i]) - static_cast<int32_t>(b[i]);
        total += static_cast<uint32_t>(diff * diff);
    }
    return total;
}

uint32_t l2_squared_sq8_avx2(const uint8_t* a, const uint8_t* b, size_t dim) noexcept {
    if (a == nullptr || b == nullptr || dim == 0) return 0;

    __m256i sum32 = _mm256_setzero_si256();
    const __m256i zero = _mm256_setzero_si256();
    size_t i = 0;

    // Process 32 uint8 coordinates per iteration
    for (; i + 32 <= dim; i += 32) {
        __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
        __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i));

        // Compute absolute difference |va - vb| using unsigned saturated subtraction
        __m256i sub_ab = _mm256_subs_epu8(va, vb);
        __m256i sub_ba = _mm256_subs_epu8(vb, va);
        __m256i diff = _mm256_or_si256(sub_ab, sub_ba);

        // Unpack 8-bit difference to 16-bit
        __m256i d_lo = _mm256_unpacklo_epi8(diff, zero);
        __m256i d_hi = _mm256_unpackhi_epi8(diff, zero);

        // Square 16-bit values and horizontally add pairs into 32-bit integers
        __m256i sq_lo = _mm256_madd_epi16(d_lo, d_lo);
        __m256i sq_hi = _mm256_madd_epi16(d_hi, d_hi);

        sum32 = _mm256_add_epi32(sum32, _mm256_add_epi32(sq_lo, sq_hi));
    }

    // Horizontal sum of 8x 32-bit integers in sum32
    __m128i lo128 = _mm256_castsi256_si128(sum32);
    __m128i hi128 = _mm256_extracti128_si256(sum32, 1);
    __m128i sum128 = _mm_add_epi32(lo128, hi128);
    sum128 = _mm_hadd_epi32(sum128, sum128);
    sum128 = _mm_hadd_epi32(sum128, sum128);
    uint32_t total = static_cast<uint32_t>(_mm_cvtsi128_si32(sum128));

    // Scalar tail for remainder
    for (; i < dim; ++i) {
        int32_t diff = static_cast<int32_t>(a[i]) - static_cast<int32_t>(b[i]);
        total += static_cast<uint32_t>(diff * diff);
    }

    return total;
}

uint32_t inner_product_sq8_scalar(const uint8_t* a, const uint8_t* b, size_t dim) noexcept {
    if (a == nullptr || b == nullptr || dim == 0) return 0;
    uint32_t total = 0;
    for (size_t i = 0; i < dim; ++i) {
        total += static_cast<uint32_t>(a[i]) * static_cast<uint32_t>(b[i]);
    }
    return total;
}

uint32_t inner_product_sq8_avx2(const uint8_t* a, const uint8_t* b, size_t dim) noexcept {
    if (a == nullptr || b == nullptr || dim == 0) return 0;

    __m256i sum32 = _mm256_setzero_si256();
    const __m256i zero = _mm256_setzero_si256();
    size_t i = 0;

    for (; i + 32 <= dim; i += 32) {
        __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
        __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i));

        __m256i a_lo = _mm256_unpacklo_epi8(va, zero);
        __m256i a_hi = _mm256_unpackhi_epi8(va, zero);
        __m256i b_lo = _mm256_unpacklo_epi8(vb, zero);
        __m256i b_hi = _mm256_unpackhi_epi8(vb, zero);

        __m256i prod_lo = _mm256_madd_epi16(a_lo, b_lo);
        __m256i prod_hi = _mm256_madd_epi16(a_hi, b_hi);

        sum32 = _mm256_add_epi32(sum32, _mm256_add_epi32(prod_lo, prod_hi));
    }

    __m128i lo128 = _mm256_castsi256_si128(sum32);
    __m128i hi128 = _mm256_extracti128_si256(sum32, 1);
    __m128i sum128 = _mm_add_epi32(lo128, hi128);
    sum128 = _mm_hadd_epi32(sum128, sum128);
    sum128 = _mm_hadd_epi32(sum128, sum128);
    uint32_t total = static_cast<uint32_t>(_mm_cvtsi128_si32(sum128));

    for (; i < dim; ++i) {
        total += static_cast<uint32_t>(a[i]) * static_cast<uint32_t>(b[i]);
    }

    return total;
}

} // namespace secan
