#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <immintrin.h>
#include <vector>

namespace secan {

/**
 * @brief FastScan 4-bit Product Quantization kernel using AVX2 in-register shuffle lookups (pshufb).
 *
 * For each subspace m in [0, M):
 * 1. The 16 centroid distance bytes fit in a 128-bit register (__m128i) and are duplicated across both lanes of __m256i.
 * 2. 32 packed bytes containing 64 4-bit codes are processed in 1 cycle using _mm256_shuffle_epi8 (VPSHUFB).
 * 3. Accumulates 16-bit intermediate distances to prevent 8-bit saturation.
 */
class FastScan4Bit {
public:
    FastScan4Bit() = default;
    FastScan4Bit(size_t dim, size_t num_subspaces);

    [[nodiscard]] size_t dim() const noexcept { return dim_; }
    [[nodiscard]] size_t M() const noexcept { return M_; }
    [[nodiscard]] size_t d_sub() const noexcept { return d_sub_; }
    static constexpr size_t K_SUB = 16; // 4-bit -> 16 centroids per subspace

    /**
     * @brief Packs 4-bit subspace codes into 32-vector interleaved blocks.
     * @param src_codes Standard row-major codes (num_vectors * M bytes, values 0..15).
     * @param dst_packed Output packed buffer.
     * @param num_vectors Number of vectors (rounded up to multiple of 32).
     */
    void pack_codes(const uint8_t* src_codes, uint8_t* dst_packed, size_t num_vectors) const;

    /**
     * @brief Computes 8-bit quantized LUT for a query.
     * Maps float distance to uint8_t [0..255] scaled by max distance.
     * Table size: M * 16 bytes.
     */
    void compute_query_lut_uint8(
        const float* query,
        const float* codebooks, // M * 16 * d_sub
        uint8_t* out_lut
    ) const;

    /**
     * @brief Scans 32 vectors simultaneously using AVX2 _mm256_shuffle_epi8.
     * @param lut_m Precomputed 16-entry LUT duplicated across both 128-bit lanes (__m256i).
     * @param packed_codes 32 packed bytes (64 4-bit codes).
     * @param out_distances_64 Output 64 bytes of distances for 32 low nibbles + 32 high nibbles.
     */
    static inline void lookup_32bytes_avx2(
        const uint8_t* packed_codes,
        __m256i lut_m,
        uint8_t* out_distances_64
    ) noexcept {
        __m256i packed_bytes = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(packed_codes));
        __m256i mask_0f = _mm256_set1_epi8(0x0F);

        __m256i low_codes = _mm256_and_si256(packed_bytes, mask_0f);
        __m256i shifted = _mm256_srli_epi16(packed_bytes, 4);
        __m256i high_codes = _mm256_and_si256(shifted, mask_0f);

        __m256i dist_low = _mm256_shuffle_epi8(lut_m, low_codes);
        __m256i dist_high = _mm256_shuffle_epi8(lut_m, high_codes);

        _mm256_storeu_si256(reinterpret_cast<__m256i*>(out_distances_64), dist_low);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(out_distances_64 + 32), dist_high);
    }

    /**
     * @brief Scans n_vectors using AVX2 in-register FastScan lookups.
     */
    void scan(
        const uint8_t* packed_codes,
        const uint8_t* lut_m_all, // M * 16 bytes
        uint16_t* out_accumulators,
        size_t num_vectors
    ) const noexcept;

private:
    size_t dim_{0};
    size_t M_{0};
    size_t d_sub_{0};
};

} // namespace secan
