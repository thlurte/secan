#include "secan/index/ivf_flat.h"
#include "secan/search/distance.h"
#include "secan/search/distance_avx2.h"
#include "secan/utils/utils.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <queue>
#include <random>
#include <stdexcept>

namespace secan {

namespace {

inline float compute_ivf_dist(const float *a, const float *b, size_t dim, MetricType metric) noexcept {
  switch (metric) {
    case MetricType::L2:
      return l2_squared_avx2_unroll4(a, b, dim);
    case MetricType::IP:
      // In min-heap ranking: higher IP means closer (smaller distance).
      return -ip_avx2_unroll4(a, b, dim);
    case MetricType::Cosine:
      return cosine_distance_avx2(a, b, dim);
  }
  return 0.0f;
}

} // anonymous namespace

void InvertedList::reserve(size_t n_vecs, size_t dim) {
  ids.reserve(n_vecs);
  data.reserve(n_vecs * dim);
}

void InvertedList::add(int32_t id, const float *vec, size_t dim) {
  ids.push_back(id);
  data.insert(data.end(), vec, vec + dim);
}

void InvertedList::clear() noexcept {
  ids.clear();
  data.clear();
}

IvfFlatIndex::IvfFlatIndex(size_t dim, size_t nlist, MetricType metric)
    : dim_(dim), nlist_(nlist), metric_(metric), is_trained_(false),
      centroids_(nlist * dim, 0.0f), lists_(nlist) {}

size_t IvfFlatIndex::total_vectors() const noexcept {
  size_t total = 0;
  for (const auto &list : lists_) {
    total += list.size();
  }
  return total;
}

InvertedListStats IvfFlatIndex::get_list_stats() const {
  InvertedListStats stats{};
  if (lists_.empty()) {
    return stats;
  }

  std::vector<size_t> sizes;
  sizes.reserve(lists_.size());

  size_t sum = 0;
  stats.min_list_size = std::numeric_limits<size_t>::max();
  stats.max_list_size = 0;

  for (const auto &list : lists_) {
    size_t sz = list.size();
    sizes.push_back(sz);
    sum += sz;
    if (sz == 0) {
      stats.empty_lists++;
    }
    if (sz < stats.min_list_size) {
      stats.min_list_size = sz;
    }
    if (sz > stats.max_list_size) {
      stats.max_list_size = sz;
    }
  }

  stats.total_vectors = sum;
  stats.mean_list_size = static_cast<double>(sum) / static_cast<double>(lists_.size());

  // Median calculation
  std::sort(sizes.begin(), sizes.end());
  size_t mid = sizes.size() / 2;
  if (sizes.size() % 2 == 0) {
    stats.median_list_size = static_cast<double>(sizes[mid - 1] + sizes[mid]) / 2.0;
  } else {
    stats.median_list_size = static_cast<double>(sizes[mid]);
  }

  // Standard deviation
  double variance_sum = 0.0;
  for (size_t sz : sizes) {
    double diff = static_cast<double>(sz) - stats.mean_list_size;
    variance_sum += diff * diff;
  }
  stats.stddev_list_size = std::sqrt(variance_sum / static_cast<double>(lists_.size()));

  return stats;
}

void IvfFlatIndex::train(size_t n, const float *data, size_t max_iters) {
  if (n < nlist_) {
    throw std::invalid_argument("Training data size must be >= nlist");
  }

  std::vector<size_t> indices(n);
  std::iota(indices.begin(), indices.end(), 0);

  std::mt19937 rng(42);
  std::shuffle(indices.begin(), indices.end(), rng);

  // Initialize centroids with random sample rows
  for (size_t k = 0; k < nlist_; ++k) {
    size_t idx = indices[k];
    std::memcpy(centroids_.data() + (k * dim_), data + (idx * dim_),
                dim_ * sizeof(float));

    if (metric_ == MetricType::Cosine || metric_ == MetricType::IP) {
      float norm_sq = 0.0f;
      for (size_t d = 0; d < dim_; ++d) {
        float val = centroids_[(k * dim_) + d];
        norm_sq += val * val;
      }
      if (norm_sq > 1e-12f) {
        float inv_norm = 1.0f / std::sqrt(norm_sq);
        for (size_t d = 0; d < dim_; ++d) {
          centroids_[(k * dim_) + d] *= inv_norm;
        }
      }
    }
  }

  for (size_t iter = 0; iter < max_iters; ++iter) {
    std::vector<float> accumulators(nlist_ * dim_, 0.0f);
    std::vector<size_t> counts(nlist_, 0);

    // Assign each vector to nearest centroid
    for (size_t i = 0; i < n; ++i) {
      const float *x = data + (i * dim_);

      size_t best_k = 0;
      float min_dist = std::numeric_limits<float>::max();

      for (size_t k = 0; k < nlist_; ++k) {
        const float *c = centroids_.data() + (k * dim_);
        float dist = compute_ivf_dist(x, c, dim_, metric_);

        if (dist < min_dist) {
          min_dist = dist;
          best_k = k;
        }
      }

      for (size_t d = 0; d < dim_; ++d) {
        accumulators[(best_k * dim_) + d] += x[d];
      }
      counts[best_k]++;
    }

    // Centroid update
    for (size_t k = 0; k < nlist_; ++k) {
      if (counts[k] > 0) {
        float norm_sq = 0.0f;
        for (size_t d = 0; d < dim_; ++d) {
          float val = accumulators[(k * dim_) + d] / static_cast<float>(counts[k]);
          centroids_[(k * dim_) + d] = val;
          norm_sq += val * val;
        }

        // Spherical normalization with exact reciprocal square root
        if ((metric_ == MetricType::Cosine || metric_ == MetricType::IP) && norm_sq > 1e-12f) {
          float inv_norm = 1.0f / std::sqrt(norm_sq);
          for (size_t d = 0; d < dim_; ++d) {
            centroids_[(k * dim_) + d] *= inv_norm;
          }
        }
      }
    }
  }

  is_trained_ = true;
}

void IvfFlatIndex::add(size_t n, const int32_t *ids, const float *data) {
  if (!is_trained_) {
    throw std::invalid_argument("Index must be trained before adding vectors.");
  }

  for (size_t i = 0; i < n; ++i) {
    const float *x = data + (i * dim_);
    int32_t id = ids[i];

    size_t best_k = 0;
    float min_dist = std::numeric_limits<float>::max();

    for (size_t k = 0; k < nlist_; ++k) {
      const float *c = centroids_.data() + (k * dim_);
      float dist = compute_ivf_dist(x, c, dim_, metric_);

      if (dist < min_dist) {
        min_dist = dist;
        best_k = k;
      }
    }

    lists_[best_k].add(id, x, dim_);
  }
}

std::vector<SearchResult> IvfFlatIndex::search(const float *query, size_t k,
                                               size_t nprobe) const {
  if (!is_trained_) {
    throw std::invalid_argument(
        "The index is not built, build before performing search.");
  }
  nprobe = std::clamp(nprobe, size_t(1), nlist_);

  std::vector<CentroidCandidate> centroid_dists(nlist_);
  for (size_t c = 0; c < nlist_; ++c) {
    const float *centroid_vec = centroids_.data() + c * dim_;
    float dist = compute_ivf_dist(query, centroid_vec, dim_, metric_);
    centroid_dists[c] = {c, dist};
  }

  std::partial_sort(centroid_dists.begin(), centroid_dists.begin() + nprobe,
                    centroid_dists.end());

  std::priority_queue<SearchResult> heap;

  for (size_t p = 0; p < nprobe; ++p) {
    size_t cluster_id = centroid_dists[p].id;
    const InvertedList &list = lists_[cluster_id];

    if (list.empty()) {
      continue;
    }

    for (size_t i = 0; i < list.size(); ++i) {
      int32_t vec_id = list.ids[i];
      const float *vec_data = list.data.data() + (i * dim_);

      float dist = compute_ivf_dist(query, vec_data, dim_, metric_);

      if (heap.size() < k) {
        heap.push(SearchResult{vec_id, dist});
      } else if (dist < heap.top().distance) {
        heap.pop();
        heap.push(SearchResult{vec_id, dist});
      }
    }
  }

  const size_t num_results = heap.size();
  std::vector<SearchResult> results(num_results);
  for (int i = static_cast<int>(num_results) - 1; i >= 0; --i) {
    results[i] = heap.top();
    heap.pop();
  }
  return results;
}

std::vector<std::vector<SearchResult>>
IvfFlatIndex::batch_search(size_t n_queries, const float *queries, size_t k,
                           size_t nprobe) const {
  std::vector<std::vector<SearchResult>> batch_results(n_queries);
  for (size_t q = 0; q < n_queries; ++q) {
    const float *query = queries + (q * dim_);
    batch_results[q] = search(query, k, nprobe);
  }
  return batch_results;
}

} // namespace secan
