#ifndef SECAN_INDEX_FLAT_H
#define SECAN_INDEX_FLAT_H

#include "secan/search/distance.h"
#include "secan/utils/io.h"
#include "secan/utils/utils.h"
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace secan {

/// Flat2DIndex: Exact brute-force nearest neighbor search index stored
/// in contiguous, cache-aligned row-major 2D floating-point memory.
class Flat2DIndex {
public:
  Flat2DIndex(size_t dim, MetricType metric = MetricType::L2);

  // Lifecycle & Vector Ingestion
  void add(size_t n, const float *vectors, const int32_t *ids = nullptr);
  void add(const FloatDataset &dataset);
  void reset() noexcept;
  void reserve(size_t n_vectors);

  // Single-query Exact Top-K Search
  [[nodiscard]] std::vector<SearchResult> search(const float *query,
                                                 size_t k) const;
  [[nodiscard]] std::vector<SearchResult>
  search(const std::vector<float> &query, size_t k) const;

  // Multi-query Tiled Batch Search (GEMV -> GEMM cache blocking)
  [[nodiscard]] std::vector<std::vector<SearchResult>>
  batch_search(size_t n_queries, const float *queries, size_t k,
               size_t tile_size = 2048) const;

  // Metadata & Accessors
  [[nodiscard]] size_t dim() const noexcept { return dim_; }
  [[nodiscard]] size_t size() const noexcept { return num_vectors_; }
  [[nodiscard]] bool empty() const noexcept { return num_vectors_ == 0; }
  [[nodiscard]] MetricType metric() const noexcept { return metric_; }
  [[nodiscard]] const float *data() const noexcept { return data_.data(); }
  [[nodiscard]] const int32_t *ids() const noexcept { return ids_.data(); }
  [[nodiscard]] const float *get(size_t idx) const noexcept {
    return data_.data() + idx * dim_;
  }
  [[nodiscard]] int32_t get_id(size_t idx) const noexcept {
    return ids_[idx];
  }

private:
  size_t dim_{0};
  size_t num_vectors_{0};
  MetricType metric_{MetricType::L2};
  std::vector<int32_t> ids_;
  std::vector<float> data_; // Flat row-major contiguous memory: N * dim floats
};

} // namespace secan

#endif // SECAN_INDEX_FLAT_H
