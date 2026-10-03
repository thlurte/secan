#ifndef SECAN_INDEX_IVF_FLAT_H
#define SECAN_INDEX_IVF_FLAT_H

#include "secan/search/distance.h"
#include "secan/utils/utils.h"
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace secan {

struct SearchResult {
  int32_t id{-1};
  float distance{0.0f};

  bool operator>(const SearchResult &other) const noexcept {
    return distance > other.distance;
  }
  bool operator<(const SearchResult &other) const noexcept {
    return distance < other.distance;
  }
};

struct CentroidCandidate {
  size_t id;
  float distance;

  bool operator<(const CentroidCandidate &other) const noexcept {
    return distance < other.distance;
  }
};

struct alignas(64) InvertedList {
  std::vector<int32_t> ids;
  std::vector<float> data;

  void reserve(size_t n_vecs, size_t dim);
  void add(int32_t id, const float *vec, size_t dim);

  [[nodiscard]] size_t size() const noexcept { return ids.size(); }

  [[nodiscard]] bool empty() const noexcept { return ids.empty(); }

  void clear() noexcept;
};

struct InvertedListStats {
  size_t total_vectors{0};
  size_t min_list_size{0};
  size_t max_list_size{0};
  double mean_list_size{0.0};
  double median_list_size{0.0};
  double stddev_list_size{0.0};
  size_t empty_lists{0};
};

class IvfFlatIndex {
public:
  IvfFlatIndex(size_t dim, size_t nlist, MetricType metric = MetricType::L2);

  // Lifecycle
  void train(size_t n, const float *data, size_t max_iters = 15);
  void add(size_t n, const int32_t *ids, const float *data);

  // Search
  [[nodiscard]] std::vector<SearchResult> search(const float *query, size_t k,
                                                 size_t nprobe = 1) const;

  [[nodiscard]] std::vector<std::vector<SearchResult>>
  batch_search(size_t n_queries, const float *queries, size_t k,
               size_t nprobe = 1) const;

  // Diagnostics & Metadata
  [[nodiscard]] InvertedListStats get_list_stats() const;
  [[nodiscard]] size_t get_dim() const noexcept { return dim_; }
  [[nodiscard]] size_t get_nlist() const noexcept { return nlist_; }
  [[nodiscard]] size_t total_vectors() const noexcept;
  [[nodiscard]] bool is_trained() const noexcept { return is_trained_; }
  [[nodiscard]] const std::vector<InvertedList> &get_lists() const noexcept {
    return lists_;
  }
  [[nodiscard]] const float *get_centroids() const noexcept {
    return centroids_.data();
  }

private:
  size_t dim_{0};
  size_t nlist_{0};
  MetricType metric_{MetricType::L2};
  bool is_trained_{false};
  std::vector<float> centroids_;    // Size: nlist * dim
  std::vector<InvertedList> lists_; // Size: nlist
};

} // namespace secan

#endif // SECAN_INDEX_IVF_FLAT_H
