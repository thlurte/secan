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
};

} // namespace secan

#endif // SECAN_QUANTIZATION_PRODUCT_QUANTIZER_H
