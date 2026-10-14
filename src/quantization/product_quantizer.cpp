#include "secan/quantization/product_quantizer.h"
#include "secan/search/distance_avx2.h"
#include <algorithm>
#include <cstring>
#include <random>
#include <stdexcept>

namespace secan {

std::vector<float> train_subspace_kmeans(
    const float *sub_data,
    size_t num_vectors,
    size_t d_sub,
    size_t k,
    size_t iters
) {
    if (num_vectors == 0 || d_sub == 0 || k == 0) {
        return {};
    }

    size_t actual_k = std::min(k, num_vectors);
    std::vector<float> centroids(k * d_sub, 0.0f);

    std::mt19937 rng(1337);
    std::uniform_int_distribution<size_t> dist(0, num_vectors - 1);

    for (size_t c = 0; c < actual_k; ++c) {
        size_t idx = dist(rng);
        std::memcpy(centroids.data() + c * d_sub, sub_data + idx * d_sub, d_sub * sizeof(float));
    }

    std::vector<float> accum(k * d_sub, 0.0f);
    std::vector<size_t> counts(k, 0);

    for (size_t iter = 0; iter < iters; ++iter) {
        std::fill(accum.begin(), accum.end(), 0.0f);
        std::fill(counts.begin(), counts.end(), 0);

        for (size_t i = 0; i < num_vectors; ++i) {
            const float *vec = sub_data + i * d_sub;
            float min_dist = 1e30f;
            size_t best_c = 0;

            for (size_t c = 0; c < actual_k; ++c) {
                float d = l2_squared_avx2_unroll4(vec, centroids.data() + c * d_sub, d_sub);
                if (d < min_dist) {
                    min_dist = d;
                    best_c = c;
                }
            }

            counts[best_c]++;
            for (size_t d = 0; d < d_sub; ++d) {
                accum[best_c * d_sub + d] += vec[d];
            }
        }

        for (size_t c = 0; c < actual_k; ++c) {
            if (counts[c] > 0) {
                float inv_n = 1.0f / static_cast<float>(counts[c]);
                for (size_t d = 0; d < d_sub; ++d) {
                    centroids[c * d_sub + d] = accum[c * d_sub + d] * inv_n;
                }
            } else {
                size_t idx = dist(rng);
                std::memcpy(centroids.data() + c * d_sub, sub_data + idx * d_sub, d_sub * sizeof(float));
            }
        }
    }

    return centroids;
}

ProductQuantizer::ProductQuantizer(size_t dimension, size_t num_subspaces)
    : dim(dimension), M(num_subspaces), d_sub(dimension / num_subspaces), k_sub(256),
      codebooks(num_subspaces * 256 * (dimension / num_subspaces), 0.0f) {
    if (num_subspaces == 0 || dimension % num_subspaces != 0) {
        throw std::invalid_argument("Dimension must be divisible by number of subspaces M");
    }
}

void ProductQuantizer::train(const float *train_data, size_t num_train, size_t iters) {
    if (num_train == 0) {
        return;
    }

    std::vector<float> sub_data(num_train * d_sub);

    for (size_t m = 0; m < M; ++m) {
        for (size_t i = 0; i < num_train; ++i) {
            std::memcpy(sub_data.data() + i * d_sub,
                        train_data + i * dim + m * d_sub,
                        d_sub * sizeof(float));
        }

        auto cb = train_subspace_kmeans(sub_data.data(), num_train, d_sub, k_sub, iters);
        std::memcpy(codebooks.data() + m * k_sub * d_sub, cb.data(), cb.size() * sizeof(float));
    }
}

void ProductQuantizer::encode(const float *src, uint8_t *dst, size_t num_vectors) const {
    for (size_t i = 0; i < num_vectors; ++i) {
        const float *vec = src + i * dim;
        uint8_t *code = dst + i * M;

        for (size_t m = 0; m < M; ++m) {
            const float *sub_vec = vec + m * d_sub;
            const float *sub_cb = codebooks.data() + m * k_sub * d_sub;

            float min_d = 1e30f;
            uint8_t best_c = 0;

            for (size_t c = 0; c < k_sub; ++c) {
                float d = l2_squared_avx2_unroll4(sub_vec, sub_cb + c * d_sub, d_sub);
                if (d < min_d) {
                    min_d = d;
                    best_c = static_cast<uint8_t>(c);
                }
            }
            code[m] = best_c;
        }
    }
}

void ProductQuantizer::decode(const uint8_t *codes, float *dst, size_t num_vectors) const {
    for (size_t i = 0; i < num_vectors; ++i) {
        const uint8_t *code = codes + i * M;
        float *vec = dst + i * dim;

        for (size_t m = 0; m < M; ++m) {
            uint8_t c = code[m];
            const float *sub_centroid = codebooks.data() + (m * k_sub + c) * d_sub;
            std::memcpy(vec + m * d_sub, sub_centroid, d_sub * sizeof(float));
        }
    }
}

float ProductQuantizer::compute_adc_distance(const float *query, const uint8_t *code) const {
    float total_dist = 0.0f;
    for (size_t m = 0; m < M; ++m) {
        const float *sub_query = query + m * d_sub;
        uint8_t c = code[m];
        const float *sub_centroid = codebooks.data() + (m * k_sub + c) * d_sub;
        total_dist += l2_squared_avx2_unroll4(sub_query, sub_centroid, d_sub);
    }
    return total_dist;
}

void ProductQuantizer::compute_query_lut(const float *query, float *out_lut) const {
    for (size_t m = 0; m < M; ++m) {
        const float *sub_query = query + m * d_sub;
        float *m_lut = out_lut + m * k_sub;
        for (size_t c = 0; c < k_sub; ++c) {
            const float *sub_centroid = codebooks.data() + (m * k_sub + c) * d_sub;
            m_lut[c] = l2_squared_avx2_unroll4(sub_query, sub_centroid, d_sub);
        }
    }
}

std::vector<float> ProductQuantizer::compute_query_lut(const float *query) const {
    std::vector<float> lut(M * k_sub);
    compute_query_lut(query, lut.data());
    return lut;
}

float ProductQuantizer::compute_adc_distance_lut(const float *lut, const uint8_t *code) const noexcept {
    float dist = 0.0f;
    for (size_t m = 0; m < M; ++m) {
        dist += lut[m * k_sub + code[m]];
    }
    return dist;
}

void ProductQuantizer::scan_adc_lut(
    const float *lut,
    const uint8_t *codes,
    float *out_distances,
    size_t n_vectors
) const noexcept {
    size_t i = 0;
    // 4-way unrolled across vectors to hide memory latency and saturate execution pipelines
    for (; i + 4 <= n_vectors; i += 4) {
        const uint8_t *c0 = codes + (i + 0) * M;
        const uint8_t *c1 = codes + (i + 1) * M;
        const uint8_t *c2 = codes + (i + 2) * M;
        const uint8_t *c3 = codes + (i + 3) * M;

        float d0 = 0.0f;
        float d1 = 0.0f;
        float d2 = 0.0f;
        float d3 = 0.0f;

        for (size_t m = 0; m < M; ++m) {
            const float *m_lut = lut + m * k_sub;
            d0 += m_lut[c0[m]];
            d1 += m_lut[c1[m]];
            d2 += m_lut[c2[m]];
            d3 += m_lut[c3[m]];
        }

        out_distances[i + 0] = d0;
        out_distances[i + 1] = d1;
        out_distances[i + 2] = d2;
        out_distances[i + 3] = d3;
    }

    // Remainder tail loop
    for (; i < n_vectors; ++i) {
        const uint8_t *c = codes + i * M;
        float d = 0.0f;
        for (size_t m = 0; m < M; ++m) {
            d += lut[m * k_sub + c[m]];
        }
        out_distances[i] = d;
    }
}

} // namespace secan
