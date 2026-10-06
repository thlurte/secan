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

} // namespace secan
