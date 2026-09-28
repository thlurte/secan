#include "secan/index/flat.h"
#include "secan/search/distance.h"
#include "secan/search/distance_avx2.h"
#include <algorithm>
#include <immintrin.h>
#include <numeric>
#include <queue>
#include <stdexcept>

namespace secan {

namespace {

inline float compute_flat_dist(MetricType metric, const float *a, const float *b, size_t dim) noexcept {
  switch (metric) {
    case MetricType::L2:
      return l2_squared_avx2_unroll4(a, b, dim);
    case MetricType::IP:
      return -ip_avx2_unroll4(a, b, dim);
    case MetricType::Cosine:
      return cosine_distance_avx2(a, b, dim);
    default:
      return l2_squared_scalar(a, b, dim);
  }
}

} // anonymous namespace

Flat2DIndex::Flat2DIndex(size_t dim, MetricType metric)
    : dim_(dim), num_vectors_(0), metric_(metric) {
  if (dim == 0) {
    throw std::invalid_argument("Flat2DIndex dimension must be > 0");
  }
}

void Flat2DIndex::reserve(size_t n_vectors) {
  ids_.reserve(n_vectors);
  data_.reserve(n_vectors * dim_);
}

void Flat2DIndex::reset() noexcept {
  num_vectors_ = 0;
  ids_.clear();
  data_.clear();
}

void Flat2DIndex::add(size_t n, const float *vectors, const int32_t *ids) {
  if (n == 0 || vectors == nullptr) {
    return;
  }

  const size_t old_size = num_vectors_;
  num_vectors_ += n;

  ids_.reserve(num_vectors_);
  if (ids != nullptr) {
    ids_.insert(ids_.end(), ids, ids + n);
  } else {
    for (size_t i = 0; i < n; ++i) {
      ids_.push_back(static_cast<int32_t>(old_size + i));
    }
  }

  data_.reserve(num_vectors_ * dim_);
  data_.insert(data_.end(), vectors, vectors + n * dim_);
}

void Flat2DIndex::add(const FloatDataset &dataset) {
  if (dataset.dim != dim_) {
    throw std::invalid_argument("Dataset dimension does not match index dimension");
  }
  add(dataset.num_vectors, dataset.data.data(), nullptr);
}

std::vector<SearchResult> Flat2DIndex::search(const float *query, size_t k) const {
  if (empty() || query == nullptr || k == 0) {
    return {};
  }

  const size_t safe_k = std::min(k, num_vectors_);
  // Max-heap to maintain the lowest distances (top-k smallest distances)
  std::priority_queue<std::pair<float, int32_t>> heap;

  for (size_t i = 0; i < num_vectors_; ++i) {
    const float *vec = data_.data() + i * dim_;
    const float dist = compute_flat_dist(metric_, query, vec, dim_);
    const int32_t id = ids_[i];

    if (heap.size() < safe_k) {
      heap.emplace(dist, id);
    } else if (dist < heap.top().first) {
      heap.pop();
      heap.emplace(dist, id);
    }
  }

  std::vector<SearchResult> results(heap.size());
  for (int i = static_cast<int>(heap.size()) - 1; i >= 0; --i) {
    results[i] = {heap.top().second, heap.top().first};
    heap.pop();
  }

  return results;
}

std::vector<SearchResult> Flat2DIndex::search(const std::vector<float> &query, size_t k) const {
  if (query.size() != dim_) {
    throw std::invalid_argument("Query dimension does not match Flat2DIndex dimension");
  }
  return search(query.data(), k);
}

std::vector<std::vector<SearchResult>>
Flat2DIndex::batch_search(size_t n_queries, const float *queries, size_t k,
                          size_t tile_size) const {
  if (n_queries == 0 || empty() || queries == nullptr || k == 0) {
    return std::vector<std::vector<SearchResult>>(n_queries);
  }

  const size_t safe_k = std::min(k, num_vectors_);
  std::vector<std::priority_queue<std::pair<float, int32_t>>> heaps(n_queries);

  // Outer loop over database tiles (streamed once from DRAM into L2/L3 cache)
  for (size_t t = 0; t < num_vectors_; t += tile_size) {
    const size_t cur_tile_size = std::min(tile_size, num_vectors_ - t);
    const float *tile_data = data_.data() + t * dim_;

    // Inner loop over queries (reusing the tile resident in cache)
    for (size_t q = 0; q < n_queries; ++q) {
      const float *q_vec = queries + q * dim_;
      auto &heap = heaps[q];

      for (size_t i = 0; i < cur_tile_size; ++i) {
        const int32_t id = ids_[t + i];
        const float *b_vec = tile_data + i * dim_;
        const float dist = compute_flat_dist(metric_, q_vec, b_vec, dim_);

        if (heap.size() < safe_k) {
          heap.emplace(dist, id);
        } else if (dist < heap.top().first) {
          heap.pop();
          heap.emplace(dist, id);
        }
      }
    }
  }

  std::vector<std::vector<SearchResult>> all_results(n_queries);
  for (size_t q = 0; q < n_queries; ++q) {
    auto &heap = heaps[q];
    all_results[q].resize(heap.size());
    for (int i = static_cast<int>(heap.size()) - 1; i >= 0; --i) {
      all_results[q][i] = {heap.top().second, heap.top().first};
      heap.pop();
    }
  }

  return all_results;
}

} // namespace secan
