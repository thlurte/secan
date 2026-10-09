#include "secan/quantization/scalar_quantizer.h"
#include <algorithm>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <numeric>

namespace secan {

void ScalarQuantizer8::train(size_t n, const float* data, float lower_pct, float upper_pct) {
    if (n == 0 || data == nullptr || dim_ == 0) {
        throw std::invalid_argument("ScalarQuantizer8::train: Invalid input data or zero dimension");
    }
    if (lower_pct < 0.0f || upper_pct > 1.0f || lower_pct >= upper_pct) {
        throw std::invalid_argument("ScalarQuantizer8::train: Invalid percentile thresholds");
    }

    size_t total_elements = n * dim_;
    size_t sample_size = std::min(total_elements, static_cast<size_t>(100000));
    
    std::vector<float> sample(data, data + sample_size);
    std::sort(sample.begin(), sample.end());

    size_t lower_idx = static_cast<size_t>(std::floor(static_cast<float>(sample.size() - 1) * lower_pct));
    size_t upper_idx = static_cast<size_t>(std::floor(static_cast<float>(sample.size() - 1) * upper_pct));

    min_val_ = sample[lower_idx];
    max_val_ = sample[upper_idx];

    // Protect against zero range (e.g. constant data)
    if (max_val_ <= min_val_) {
        max_val_ = min_val_ + 1e-5f;
    }

    step_ = (max_val_ - min_val_) / 255.0f;
    inv_step_ = 1.0f / step_;
    trained_ = true;
}

void ScalarQuantizer8::encode(const float* src, uint8_t* dst, size_t n_vectors) const {
    if (!trained_) {
        throw std::runtime_error("ScalarQuantizer8::encode: Quantizer has not been trained yet");
    }
    if (src == nullptr || dst == nullptr) {
        return;
    }

    size_t total_elements = n_vectors * dim_;
    for (size_t i = 0; i < total_elements; ++i) {
        float val = std::clamp(src[i], min_val_, max_val_);
        float scaled = (val - min_val_) * inv_step_;
        dst[i] = static_cast<uint8_t>(std::clamp(std::round(scaled), 0.0f, 255.0f));
    }
}

void ScalarQuantizer8::decode(const uint8_t* src, float* dst, size_t n_vectors) const {
    if (!trained_) {
        throw std::runtime_error("ScalarQuantizer8::decode: Quantizer has not been trained yet");
    }
    if (src == nullptr || dst == nullptr) {
        return;
    }

    size_t total_elements = n_vectors * dim_;
    for (size_t i = 0; i < total_elements; ++i) {
        dst[i] = min_val_ + static_cast<float>(src[i]) * step_;
    }
}

double ScalarQuantizer8::compute_mse(const float* orig, const float* recon, size_t count) {
    if (orig == nullptr || recon == nullptr || count == 0) {
        return 0.0;
    }
    double sum_sq_diff = 0.0;
    for (size_t i = 0; i < count; ++i) {
        double diff = static_cast<double>(orig[i]) - static_cast<double>(recon[i]);
        sum_sq_diff += diff * diff;
    }
    return sum_sq_diff / static_cast<double>(count);
}

// ============================================================================
// ScalarQuantizer4 Implementation
// ============================================================================

void ScalarQuantizer4::train(size_t n, const float* data, float lower_pct, float upper_pct) {
    if (n == 0 || data == nullptr || dim_ == 0) {
        throw std::invalid_argument("ScalarQuantizer4::train: Invalid input data or zero dimension");
    }
    if (lower_pct < 0.0f || upper_pct > 1.0f || lower_pct >= upper_pct) {
        throw std::invalid_argument("ScalarQuantizer4::train: Invalid percentile thresholds");
    }

    size_t total_elements = n * dim_;
    size_t sample_size = std::min(total_elements, static_cast<size_t>(100000));
    
    std::vector<float> sample(data, data + sample_size);
    std::sort(sample.begin(), sample.end());

    size_t lower_idx = static_cast<size_t>(std::floor(static_cast<float>(sample.size() - 1) * lower_pct));
    size_t upper_idx = static_cast<size_t>(std::floor(static_cast<float>(sample.size() - 1) * upper_pct));

    min_val_ = sample[lower_idx];
    max_val_ = sample[upper_idx];

    // Protect against zero range (e.g. constant data)
    if (max_val_ <= min_val_) {
        max_val_ = min_val_ + 1e-5f;
    }

    // 4-bit scalar quantizer has 15 discrete steps [0..15]
    step_ = (max_val_ - min_val_) / 15.0f;
    inv_step_ = 1.0f / step_;
    trained_ = true;
}

void ScalarQuantizer4::encode(const float* src, uint8_t* dst, size_t n_vectors) const {
    if (!trained_) {
        throw std::runtime_error("ScalarQuantizer4::encode: Quantizer has not been trained yet");
    }
    if (src == nullptr || dst == nullptr) {
        return;
    }

    size_t bytes_per_vec = bytes_per_vector();
    for (size_t v = 0; v < n_vectors; ++v) {
        const float* src_vec = src + v * dim_;
        uint8_t* dst_vec = dst + v * bytes_per_vec;

        size_t d = 0;
        for (; d + 1 < dim_; d += 2) {
            float v0 = std::clamp(src_vec[d], min_val_, max_val_);
            float v1 = std::clamp(src_vec[d + 1], min_val_, max_val_);

            uint8_t q0 = static_cast<uint8_t>(std::clamp(std::round((v0 - min_val_) * inv_step_), 0.0f, 15.0f)) & 0x0F;
            uint8_t q1 = static_cast<uint8_t>(std::clamp(std::round((v1 - min_val_) * inv_step_), 0.0f, 15.0f)) & 0x0F;

            dst_vec[d / 2] = static_cast<uint8_t>((q1 << 4) | q0);
        }

        // Handle odd remainder dimension if any
        if (d < dim_) {
            float v0 = std::clamp(src_vec[d], min_val_, max_val_);
            uint8_t q0 = static_cast<uint8_t>(std::clamp(std::round((v0 - min_val_) * inv_step_), 0.0f, 15.0f)) & 0x0F;
            dst_vec[d / 2] = q0;
        }
    }
}

void ScalarQuantizer4::decode(const uint8_t* src, float* dst, size_t n_vectors) const {
    if (!trained_) {
        throw std::runtime_error("ScalarQuantizer4::decode: Quantizer has not been trained yet");
    }
    if (src == nullptr || dst == nullptr) {
        return;
    }

    size_t bytes_per_vec = bytes_per_vector();
    for (size_t v = 0; v < n_vectors; ++v) {
        const uint8_t* src_vec = src + v * bytes_per_vec;
        float* dst_vec = dst + v * dim_;

        size_t d = 0;
        for (; d + 1 < dim_; d += 2) {
            uint8_t byte = src_vec[d / 2];
            uint8_t q0 = byte & 0x0F;
            uint8_t q1 = (byte >> 4) & 0x0F;

            dst_vec[d] = min_val_ + static_cast<float>(q0) * step_;
            dst_vec[d + 1] = min_val_ + static_cast<float>(q1) * step_;
        }

        if (d < dim_) {
            uint8_t q0 = src_vec[d / 2] & 0x0F;
            dst_vec[d] = min_val_ + static_cast<float>(q0) * step_;
        }
    }
}

float ScalarQuantizer4::compute_distance_l2(const float* query, const uint8_t* packed_base) const noexcept {
    float total_dist = 0.0f;
    size_t d = 0;
    for (; d + 1 < dim_; d += 2) {
        uint8_t byte = packed_base[d / 2];
        uint8_t q0 = byte & 0x0F;
        uint8_t q1 = (byte >> 4) & 0x0F;

        float rec0 = min_val_ + static_cast<float>(q0) * step_;
        float rec1 = min_val_ + static_cast<float>(q1) * step_;

        float diff0 = query[d] - rec0;
        float diff1 = query[d + 1] - rec1;

        total_dist += diff0 * diff0 + diff1 * diff1;
    }
    if (d < dim_) {
        uint8_t q0 = packed_base[d / 2] & 0x0F;
        float rec0 = min_val_ + static_cast<float>(q0) * step_;
        float diff0 = query[d] - rec0;
        total_dist += diff0 * diff0;
    }
    return total_dist;
}

void ScalarQuantizer4::compute_distance_l2_batch(
    const float* query, const uint8_t* packed_base, float* out_distances, size_t n_vectors) const noexcept {
    size_t bytes_per_vec = bytes_per_vector();
    for (size_t i = 0; i < n_vectors; ++i) {
        out_distances[i] = compute_distance_l2(query, packed_base + i * bytes_per_vec);
    }
}

double ScalarQuantizer4::compute_mse(const float* orig, const float* recon, size_t count) {
    return ScalarQuantizer8::compute_mse(orig, recon, count);
}

} // namespace secan

