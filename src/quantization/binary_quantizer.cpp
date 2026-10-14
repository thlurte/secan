#include "secan/quantization/binary_quantizer.h"
#include <cstring>
#include <immintrin.h>

namespace secan {

BinaryQuantizer::BinaryQuantizer(size_t dim) : dim_(dim) {
    if (dim == 0) {
        throw std::invalid_argument("BinaryQuantizer: Dimension must be greater than 0");
    }
    // Round up to nearest multiple of 64
    num_u64_ = (dim_ + 63) / 64;
}

void BinaryQuantizer::encode(const float* src, uint64_t* dst, size_t n_vectors) const {
    if (src == nullptr || dst == nullptr || n_vectors == 0) {
        return;
    }

    std::memset(dst, 0, n_vectors * num_u64_ * sizeof(uint64_t));

    for (size_t v = 0; v < n_vectors; ++v) {
        const float* src_vec = src + v * dim_;
        uint64_t* dst_vec = dst + v * num_u64_;

        for (size_t d = 0; d < dim_; ++d) {
            if (src_vec[d] >= 0.0f) {
                size_t word_idx = d / 64;
                size_t bit_idx = d % 64;
                dst_vec[word_idx] |= (1ULL << bit_idx);
            }
        }
    }
}

void BinaryQuantizer::decode(const uint64_t* src, float* dst, size_t n_vectors) const {
    if (src == nullptr || dst == nullptr || n_vectors == 0) {
        return;
    }

    for (size_t v = 0; v < n_vectors; ++v) {
        const uint64_t* src_vec = src + v * num_u64_;
        float* dst_vec = dst + v * dim_;

        for (size_t d = 0; d < dim_; ++d) {
            size_t word_idx = d / 64;
            size_t bit_idx = d % 64;
            bool bit_set = (src_vec[word_idx] >> bit_idx) & 1ULL;
            dst_vec[d] = bit_set ? 1.0f : -1.0f;
        }
    }
}

void BinaryQuantizer::compute_distance_ip_batch(
    const float* __restrict__ query,
    const uint64_t* __restrict__ packed_base,
    float* __restrict__ out_distances,
    size_t n_vectors
) const noexcept {
    // Precompute query sum once for all database vectors
    float query_sum = 0.0f;
    for (size_t d = 0; d < dim_; ++d) {
        query_sum += query[d];
    }

    for (size_t i = 0; i < n_vectors; ++i) {
        out_distances[i] = asymmetric_ip_bq(
            query,
            packed_base + i * num_u64_,
            num_u64_,
            query_sum
        );
    }
}

uint32_t BinaryQuantizer::hamming_distance(const uint64_t* a, const uint64_t* b, size_t num_u64) noexcept {
    uint32_t dist = 0;
    for (size_t i = 0; i < num_u64; ++i) {
        uint64_t diff = a[i] ^ b[i];
        dist += static_cast<uint32_t>(__builtin_popcountll(diff));
    }
    return dist;
}

} // namespace secan
