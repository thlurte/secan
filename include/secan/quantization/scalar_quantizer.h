#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace secan {

/**
 * @brief 8-bit Scalar Quantizer (SQ8) with Percentile-Based Outlier Clipping.
 * 
 * Maps continuous 32-bit floating point vectors into compact 8-bit unsigned integers (uint8_t),
 * reducing memory bandwidth and footprint by exactly 4x.
 */
class ScalarQuantizer8 {
public:
    ScalarQuantizer8() = default;
    explicit ScalarQuantizer8(size_t dim) : dim_(dim) {}

    /**
     * @brief Trains quantization bounds using percentile clipping.
     * @param n Number of vectors.
     * @param data Pointer to contiguous float32 vector data (n * dim).
     * @param lower_pct Lower clipping percentile (default 0.05th percentile = 0.0005).
     * @param upper_pct Upper clipping percentile (default 99.95th percentile = 0.9995).
     */
    void train(size_t n, const float* data, float lower_pct = 0.0005f, float upper_pct = 0.9995f);

    /**
     * @brief Encodes float32 vectors into uint8_t quantized representation.
     * @param src Input float32 vectors (n_vectors * dim).
     * @param dst Output uint8_t buffer (n_vectors * dim).
     * @param n_vectors Number of vectors to encode.
     */
    void encode(const float* src, uint8_t* dst, size_t n_vectors) const;

    /**
     * @brief Decodes uint8_t quantized representation back into approximate float32 vectors.
     * @param src Input uint8_t quantized buffer (n_vectors * dim).
     * @param dst Output float32 vectors (n_vectors * dim).
     * @param n_vectors Number of vectors to decode.
     */
    void decode(const uint8_t* src, float* dst, size_t n_vectors) const;

    [[nodiscard]] size_t dim() const noexcept { return dim_; }
    [[nodiscard]] float min_val() const noexcept { return min_val_; }
    [[nodiscard]] float max_val() const noexcept { return max_val_; }
    [[nodiscard]] float step() const noexcept { return step_; }
    [[nodiscard]] float inv_step() const noexcept { return inv_step_; }
    [[nodiscard]] bool is_trained() const noexcept { return trained_; }

    /**
     * @brief Computes Mean Squared Error (MSE) between original and reconstructed float data.
     */
    static double compute_mse(const float* orig, const float* recon, size_t count);

private:
    size_t dim_{0};
    float min_val_{0.0f};
    float max_val_{0.0f};
    float step_{0.0f};
    float inv_step_{0.0f};
    bool trained_{false};
};

/**
 * @brief 4-bit Scalar Quantizer (SQ4) with Nibble Packing.
 * 
 * Compresses continuous 32-bit floating point vectors into 4-bit unsigned integers (nibbles),
 * packing two 4-bit values per uint8_t byte. Compresses 128-D vectors from 512 bytes to exactly 64 bytes
 * (fitting perfectly into a single L1 cache line with 8x compression).
 */
class ScalarQuantizer4 {
public:
    ScalarQuantizer4() = default;
    explicit ScalarQuantizer4(size_t dim) : dim_(dim) {}

    /**
     * @brief Trains quantization bounds using percentile clipping.
     * @param n Number of vectors.
     * @param data Pointer to contiguous float32 vector data (n * dim).
     * @param lower_pct Lower clipping percentile (default 0.05th percentile = 0.0005).
     * @param upper_pct Upper clipping percentile (default 99.95th percentile = 0.9995).
     */
    void train(size_t n, const float* data, float lower_pct = 0.0005f, float upper_pct = 0.9995f);

    /**
     * @brief Encodes float32 vectors into packed 4-bit representation (dim / 2 bytes per vector).
     * @param src Input float32 vectors (n_vectors * dim).
     * @param dst Output packed uint8_t buffer (n_vectors * bytes_per_vector()).
     * @param n_vectors Number of vectors to encode.
     */
    void encode(const float* src, uint8_t* dst, size_t n_vectors) const;

    /**
     * @brief Decodes packed 4-bit representation back into approximate float32 vectors.
     * @param src Input packed uint8_t buffer (n_vectors * bytes_per_vector()).
     * @param dst Output float32 vectors (n_vectors * dim).
     * @param n_vectors Number of vectors to decode.
     */
    void decode(const uint8_t* src, float* dst, size_t n_vectors) const;

    /**
     * @brief Computes approximate squared L2 distance between float32 query and packed 4-bit vector.
     */
    [[nodiscard]] float compute_distance_l2(const float* query, const uint8_t* packed_base) const noexcept;

    /**
     * @brief Batched approximate squared L2 distance computation.
     */
    void compute_distance_l2_batch(const float* query, const uint8_t* packed_base, float* out_distances, size_t n_vectors) const noexcept;

    [[nodiscard]] size_t dim() const noexcept { return dim_; }
    [[nodiscard]] size_t bytes_per_vector() const noexcept { return (dim_ + 1) / 2; }
    [[nodiscard]] float min_val() const noexcept { return min_val_; }
    [[nodiscard]] float max_val() const noexcept { return max_val_; }
    [[nodiscard]] float step() const noexcept { return step_; }
    [[nodiscard]] float inv_step() const noexcept { return inv_step_; }
    [[nodiscard]] bool is_trained() const noexcept { return trained_; }

    /**
     * @brief Computes Mean Squared Error (MSE) between original and reconstructed float data.
     */
    static double compute_mse(const float* orig, const float* recon, size_t count);

private:
    size_t dim_{0};
    float min_val_{0.0f};
    float max_val_{0.0f};
    float step_{0.0f};
    float inv_step_{0.0f};
    bool trained_{false};
};

} // namespace secan

