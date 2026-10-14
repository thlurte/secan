#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace secan {

/**
 * @brief Computes asymmetric inner product between FP32 query and 1-bit packed binary vector.
 *
 * Mathematical formulation:
 * Each bit b_d in {0, 1} represents signed coordinate x_hat_d = 2 * b_d - 1 in {-1, +1}.
 * <q, x_hat> = sum(q_d * (2*b_d - 1)) = 2 * sum_{d: b_d = 1}(q_d) - sum(q_d).
 * 
 * Uses __builtin_ctzll and Brian Kernighan's bit-clearing to eliminate floating-point multiplications.
 *
 * @param query Pointer to contiguous FP32 query vector.
 * @param bq_bits Pointer to packed uint64_t binary words.
 * @param num_u64 Number of 64-bit words (dim / 64).
 * @param query_sum Precomputed scalar sum of all elements in query: sum(q_d).
 * @return Inner product between FP32 query and signed binary vector.
 */
inline float asymmetric_ip_bq(
    const float* __restrict__ query,
    const uint64_t* __restrict__ bq_bits,
    size_t num_u64,
    float query_sum
) noexcept {
    float pos_sum = 0.0f;
    for (size_t w = 0; w < num_u64; ++w) {
        uint64_t word = bq_bits[w];
        const float* q_word = query + w * 64;

        // Bitscan through all set bits
        while (word != 0) {
            int bit = __builtin_ctzll(word);
            pos_sum += q_word[bit];
            word &= (word - 1); // Clear lowest set bit
        }
    }
    return 2.0f * pos_sum - query_sum;
}

/**
 * @brief 1-Bit Binary Quantizer (BQ).
 *
 * Compresses D-dimensional FP32 vectors into D bits packed into 64-bit words (uint64_t).
 * Achieves 32x memory compression (128-D vector = 16 bytes).
 */
class BinaryQuantizer {
public:
    BinaryQuantizer() = default;
    explicit BinaryQuantizer(size_t dim);

    /**
     * @brief Encodes float32 vectors into 1-bit packed binary representation.
     * @param src Input float32 vectors (n_vectors * dim).
     * @param dst Output packed uint64_t buffer (n_vectors * num_u64).
     * @param n_vectors Number of vectors to encode.
     */
    void encode(const float* src, uint64_t* dst, size_t n_vectors) const;

    /**
     * @brief Decodes 1-bit binary vectors into approximate signed float32 vectors (-1.0f or +1.0f).
     * @param src Input packed uint64_t buffer (n_vectors * num_u64).
     * @param dst Output float32 vectors (n_vectors * dim).
     * @param n_vectors Number of vectors to decode.
     */
    void decode(const uint64_t* src, float* dst, size_t n_vectors) const;

    /**
     * @brief Evaluates asymmetric inner product between single FP32 query and single packed vector.
     */
    [[nodiscard]] float compute_distance_ip(
        const float* __restrict__ query,
        const uint64_t* __restrict__ bq_bits,
        float query_sum
    ) const noexcept {
        return asymmetric_ip_bq(query, bq_bits, num_u64_, query_sum);
    }

    /**
     * @brief Batch evaluates asymmetric inner product against n_vectors packed vectors.
     */
    void compute_distance_ip_batch(
        const float* __restrict__ query,
        const uint64_t* __restrict__ packed_base,
        float* __restrict__ out_distances,
        size_t n_vectors
    ) const noexcept;

    /**
     * @brief Computes hardware-accelerated Hamming distance between two packed binary vectors.
     */
    static uint32_t hamming_distance(const uint64_t* a, const uint64_t* b, size_t num_u64) noexcept;

    [[nodiscard]] size_t dim() const noexcept { return dim_; }
    [[nodiscard]] size_t num_u64() const noexcept { return num_u64_; }
    [[nodiscard]] size_t bytes_per_vector() const noexcept { return num_u64_ * sizeof(uint64_t); }

private:
    size_t dim_{0};
    size_t num_u64_{0};
};

} // namespace secan
