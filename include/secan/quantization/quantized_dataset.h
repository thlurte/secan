#pragma once

#include <span>
#include <vector>
#include <cstdint>
#include <cstddef>
#include <stdexcept>

namespace secan {

/**
 * @brief Zero-Copy 8-Bit Quantized Dataset View.
 * 
 * Provides lightweight non-owning std::span views over contiguous 8-bit quantized memory buffers.
 * Guarantees zero dynamic heap allocations during high-throughput search traversals.
 */
class QuantizedDatasetSQ8 {
public:
    QuantizedDatasetSQ8() = default;

    QuantizedDatasetSQ8(size_t num_vectors, size_t dim, std::vector<uint8_t> data)
        : n_(num_vectors), dim_(dim), data_(std::move(data)), raw_ptr_(data_.data()) {
        if (raw_ptr_ == nullptr && n_ > 0) {
            throw std::invalid_argument("QuantizedDatasetSQ8: data buffer is empty");
        }
    }

    QuantizedDatasetSQ8(size_t num_vectors, size_t dim, const uint8_t* raw_ptr)
        : n_(num_vectors), dim_(dim), raw_ptr_(raw_ptr) {
        if (raw_ptr_ == nullptr && n_ > 0) {
            throw std::invalid_argument("QuantizedDatasetSQ8: null raw pointer provided");
        }
    }

    [[nodiscard]] std::span<const uint8_t> get_vector(size_t idx) const noexcept {
        return std::span<const uint8_t>(raw_ptr_ + idx * dim_, dim_);
    }

    [[nodiscard]] std::span<const uint8_t> get_block(size_t start_idx, size_t count) const noexcept {
        return std::span<const uint8_t>(raw_ptr_ + start_idx * dim_, count * dim_);
    }

    [[nodiscard]] const uint8_t* data() const noexcept { return raw_ptr_; }
    [[nodiscard]] size_t size() const noexcept { return n_; }
    [[nodiscard]] size_t dim() const noexcept { return dim_; }
    [[nodiscard]] size_t nbytes() const noexcept { return n_ * dim_; }

private:
    size_t n_{0};
    size_t dim_{0};
    std::vector<uint8_t> data_;
    const uint8_t* raw_ptr_{nullptr};
};

/**
 * @brief Zero-Copy 4-Bit Packed Quantized Dataset View.
 * 
 * Provides std::span views over 4-bit packed nibble buffers ((dim + 1) / 2 bytes per vector).
 */
class QuantizedDatasetSQ4 {
public:
    QuantizedDatasetSQ4() = default;

    QuantizedDatasetSQ4(size_t num_vectors, size_t dim, std::vector<uint8_t> data)
        : n_(num_vectors), dim_(dim), bytes_per_vec_((dim + 1) / 2), data_(std::move(data)), raw_ptr_(data_.data()) {}

    QuantizedDatasetSQ4(size_t num_vectors, size_t dim, const uint8_t* raw_ptr)
        : n_(num_vectors), dim_(dim), bytes_per_vec_((dim + 1) / 2), raw_ptr_(raw_ptr) {}

    [[nodiscard]] std::span<const uint8_t> get_vector(size_t idx) const noexcept {
        return std::span<const uint8_t>(raw_ptr_ + idx * bytes_per_vec_, bytes_per_vec_);
    }

    [[nodiscard]] std::span<const uint8_t> get_block(size_t start_idx, size_t count) const noexcept {
        return std::span<const uint8_t>(raw_ptr_ + start_idx * bytes_per_vec_, count * bytes_per_vec_);
    }

    [[nodiscard]] const uint8_t* data() const noexcept { return raw_ptr_; }
    [[nodiscard]] size_t size() const noexcept { return n_; }
    [[nodiscard]] size_t dim() const noexcept { return dim_; }
    [[nodiscard]] size_t bytes_per_vector() const noexcept { return bytes_per_vec_; }
    [[nodiscard]] size_t nbytes() const noexcept { return n_ * bytes_per_vec_; }

private:
    size_t n_{0};
    size_t dim_{0};
    size_t bytes_per_vec_{0};
    std::vector<uint8_t> data_;
    const uint8_t* raw_ptr_{nullptr};
};

} // namespace secan
