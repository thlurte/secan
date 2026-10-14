#include "secan/quantization/fastscan.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace secan {

FastScan4Bit::FastScan4Bit(size_t dim, size_t num_subspaces)
    : dim_(dim), M_(num_subspaces), d_sub_(dim / (num_subspaces ? num_subspaces : 1)) {
    if (num_subspaces == 0 || dim == 0 || dim % num_subspaces != 0) {
        throw std::invalid_argument("FastScan4Bit: dim must be positive and divisible by num_subspaces");
    }
}

void FastScan4Bit::pack_codes(
    const uint8_t* src_codes,
    uint8_t* dst_packed,
    size_t num_vectors
) const {
    if (num_vectors == 0 || M_ == 0) return;

    size_t num_blocks = (num_vectors + 31) / 32;

    for (size_t b = 0; b < num_blocks; ++b) {
        uint8_t* block_dst = dst_packed + b * M_ * 16;
        for (size_t m = 0; m < M_; ++m) {
            uint8_t* m_dst = block_dst + m * 16;
            for (size_t j = 0; j < 16; ++j) {
                size_t v_low = b * 32 + j;
                size_t v_high = b * 32 + 16 + j;
                uint8_t c_low = (v_low < num_vectors) ? (src_codes[v_low * M_ + m] & 0x0F) : 0;
                uint8_t c_high = (v_high < num_vectors) ? (src_codes[v_high * M_ + m] & 0x0F) : 0;
                m_dst[j] = static_cast<uint8_t>(c_low | (c_high << 4));
            }
        }
    }
}

void FastScan4Bit::compute_query_lut_uint8(
    const float* query,
    const float* codebooks,
    uint8_t* out_lut
) const {
    if (M_ == 0 || d_sub_ == 0) return;

    std::vector<float> raw_distances(M_ * K_SUB);

    float min_d = 1e30f;
    float max_d = -1e30f;

    for (size_t m = 0; m < M_; ++m) {
        const float* subquery = query + m * d_sub_;
        for (size_t k = 0; k < K_SUB; ++k) {
            const float* centroid = codebooks + (m * K_SUB + k) * d_sub_;
            float sum_sq = 0.0f;
            for (size_t d = 0; d < d_sub_; ++d) {
                float diff = subquery[d] - centroid[d];
                sum_sq += diff * diff;
            }
            raw_distances[m * K_SUB + k] = sum_sq;
            min_d = std::min(min_d, sum_sq);
            max_d = std::max(max_d, sum_sq);
        }
    }

    if (max_d - min_d < 1e-7f) {
        std::fill(out_lut, out_lut + M_ * K_SUB, static_cast<uint8_t>(0));
        return;
    }

    float scale = 255.0f / (max_d - min_d);
    for (size_t i = 0; i < M_ * K_SUB; ++i) {
        float scaled = (raw_distances[i] - min_d) * scale;
        out_lut[i] = static_cast<uint8_t>(std::clamp(std::round(scaled), 0.0f, 255.0f));
    }
}

void FastScan4Bit::scan(
    const uint8_t* packed_codes,
    const uint8_t* lut_m_all,
    uint16_t* out_accumulators,
    size_t num_vectors
) const noexcept {
    if (num_vectors == 0 || M_ == 0) return;

    size_t num_blocks = (num_vectors + 31) / 32;
    const __m128i mask_0f = _mm_set1_epi8(0x0F);

    for (size_t b = 0; b < num_blocks; ++b) {
        const uint8_t* block_packed = packed_codes + b * M_ * 16;
        __m256i acc0 = _mm256_setzero_si256();
        __m256i acc1 = _mm256_setzero_si256();

        for (size_t m = 0; m < M_; ++m) {
            __m128i lut_m = _mm_loadu_si128(reinterpret_cast<const __m128i*>(lut_m_all + m * 16));
            __m128i codes = _mm_loadu_si128(reinterpret_cast<const __m128i*>(block_packed + m * 16));

            __m128i low_codes = _mm_and_si128(codes, mask_0f);
            __m128i high_codes = _mm_and_si128(_mm_srli_epi16(codes, 4), mask_0f);

            __m128i d_low = _mm_shuffle_epi8(lut_m, low_codes);
            __m128i d_high = _mm_shuffle_epi8(lut_m, high_codes);

            acc0 = _mm256_add_epi16(acc0, _mm256_cvtepu8_epi16(d_low));
            acc1 = _mm256_add_epi16(acc1, _mm256_cvtepu8_epi16(d_high));
        }

        size_t base_vec = b * 32;
        if (base_vec + 32 <= num_vectors) {
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(out_accumulators + base_vec), acc0);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(out_accumulators + base_vec + 16), acc1);
        } else {
            alignas(32) uint16_t tmp0[16];
            alignas(32) uint16_t tmp1[16];
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(tmp0), acc0);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(tmp1), acc1);
            for (size_t i = 0; i < 16 && (base_vec + i) < num_vectors; ++i) {
                out_accumulators[base_vec + i] = tmp0[i];
            }
            for (size_t i = 0; i < 16 && (base_vec + 16 + i) < num_vectors; ++i) {
                out_accumulators[base_vec + 16 + i] = tmp1[i];
            }
        }
    }
}

} // namespace secan
