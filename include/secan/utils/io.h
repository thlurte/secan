#ifndef SECAN_UTILS_IO_H
#define SECAN_UTILS_IO_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace secan {

/// Row-major contiguous dataset of single-precision floating point vectors.
struct FloatDataset {
  size_t num_vectors{0};
  size_t dim{0};
  std::vector<float> data; // Flat row-major contiguous memory: N * D floats

  [[nodiscard]] const float *get(size_t index) const noexcept {
    return data.data() + index * dim;
  }

  [[nodiscard]] float *get(size_t index) noexcept {
    return data.data() + index * dim;
  }

  [[nodiscard]] size_t size() const noexcept { return num_vectors; }
  [[nodiscard]] bool empty() const noexcept { return num_vectors == 0; }
};

/// Row-major contiguous dataset of unsigned 8-bit byte vectors.
struct ByteDataset {
  size_t num_vectors{0};
  size_t dim{0};
  std::vector<uint8_t> data; // Flat row-major contiguous memory: N * D bytes

  [[nodiscard]] const uint8_t *get(size_t index) const noexcept {
    return data.data() + index * dim;
  }

  [[nodiscard]] uint8_t *get(size_t index) noexcept {
    return data.data() + index * dim;
  }

  [[nodiscard]] size_t size() const noexcept { return num_vectors; }
  [[nodiscard]] bool empty() const noexcept { return num_vectors == 0; }
};

/// Row-major contiguous dataset of 32-bit signed integer vectors (e.g. ground truth k-NN indices).
struct IntDataset {
  size_t num_vectors{0};
  size_t dim{0};
  std::vector<int32_t> data; // Flat row-major contiguous memory: N * D ints

  [[nodiscard]] const int32_t *get(size_t index) const noexcept {
    return data.data() + index * dim;
  }

  [[nodiscard]] int32_t *get(size_t index) noexcept {
    return data.data() + index * dim;
  }

  [[nodiscard]] size_t size() const noexcept { return num_vectors; }
  [[nodiscard]] bool empty() const noexcept { return num_vectors == 0; }
};

/// Zero-copy memory-mapped view of an .fvecs file on disk.
class MmapFloatDataset {
public:
  MmapFloatDataset() = default;
  ~MmapFloatDataset();

  // Move-only semantics
  MmapFloatDataset(const MmapFloatDataset &) = delete;
  MmapFloatDataset &operator=(const MmapFloatDataset &) = delete;
  MmapFloatDataset(MmapFloatDataset &&other) noexcept;
  MmapFloatDataset &operator=(MmapFloatDataset &&other) noexcept;

  [[nodiscard]] const float *get(size_t index) const noexcept;
  [[nodiscard]] size_t size() const noexcept { return num_vectors_; }
  [[nodiscard]] size_t dim() const noexcept { return dim_; }
  [[nodiscard]] bool empty() const noexcept { return num_vectors_ == 0; }
  [[nodiscard]] bool is_open() const noexcept { return mapped_data_ != nullptr; }

  void close();

  // Factory loader
  static MmapFloatDataset open(const std::string &path);

private:
  void *mapped_data_{nullptr};
  size_t file_size_{0};
  size_t num_vectors_{0};
  size_t dim_{0};
  size_t record_size_{0};
  int fd_{-1};
};

/// Binary file readers (std::ifstream buffered read)
FloatDataset load_fvecs(const std::string &path);
ByteDataset load_bvecs(const std::string &path);
IntDataset load_ivecs(const std::string &path);

/// Memory-mapped zero-copy reader
MmapFloatDataset load_fvecs_mmap(const std::string &path);

/// Binary file writers (useful for test generation, preprocessing, and exports)
void write_fvecs(const std::string &path, const FloatDataset &ds);
void write_bvecs(const std::string &path, const ByteDataset &ds);
void write_ivecs(const std::string &path, const IntDataset &ds);

} // namespace secan

#endif // SECAN_UTILS_IO_H
