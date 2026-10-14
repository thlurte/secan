#ifndef SECAN_QUANTIZATION_PRODUCT_QUANTIZER_H
#define SECAN_QUANTIZATION_PRODUCT_QUANTIZER_H

#include <cstddef>
#include <cstdint>
#include <vector>

namespace secan {

std::vector<float> train_subspace_kmeans(
    const float *sub_data,
    size_t num_vectors,
    size_t d_sub,
    size_t k,
    size_t iters = 15
);

class ProductQuantizer {
public:
    size_t dim{0};
    size_t M{0};
    size_t d_sub{0};
    size_t k_sub{256};
    std::vector<float> codebooks; // M * k_sub * d_sub floats

    ProductQuantizer() = default;
    ProductQuantizer(size_t dimension, size_t num_subspaces);

    void train(const float *train_data, size_t num_train, size_t iters = 15);

    void encode(const float *src, uint8_t *dst, size_t num_vectors) const;

    void decode(const uint8_t *codes, float *dst, size_t num_vectors) const;

    float compute_adc_distance(const float *query, const uint8_t *code) const;

    /**
     * @brief Precomputes distance between query subvectors and all centroids in each subspace.
     * Shape: M * k_sub floats (for M=8, k_sub=256: 2048 floats = 8 KB, L1D resident).
     * @param query Input float32 query vector of size dim.
     * @param out_lut Pre-allocated buffer of size M * k_sub.
     */
    void compute_query_lut(const float *query, float *out_lut) const;
    std::vector<float> compute_query_lut(const float *query) const;

    /**
     * @brief Evaluates asymmetric distance using precomputed LUT in O(M) table lookups.
     */
    float compute_adc_distance_lut(const float *lut, const uint8_t *code) const noexcept;

    /**
     * @brief High-throughput 4-way unrolled dataset scan using precomputed query LUT.
     */
    void scan_adc_lut(
        const float *lut,
        const uint8_t *codes,
        float *out_distances,
        size_t n_vectors
    ) const noexcept;
};

} // namespace secan

#endif // SECAN_QUANTIZATION_PRODUCT_QUANTIZER_H
