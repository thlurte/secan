#include "secan/search/search.h"
#include "secan/search/distance.h"
#include "secan/search/distance_avx2.h"
#include <algorithm>
#include <immintrin.h>
#include <queue>
#include <stdexcept>

namespace secan {

namespace {

inline MetricType parse_metric(const std::string &method) {
  if (method == "cosine") {
    return MetricType::Cosine;
  } else if (method == "ip" || method == "inner_product" || method == "dot") {
    return MetricType::IP;
  }
  return MetricType::L2;
}

inline float compute_distance(MetricType metric, const float *row, const float *query, size_t dim) {
  switch (metric) {
    case MetricType::Cosine:
      return cosine_distance_scalar(row, query, dim);
    case MetricType::IP:
      // In MIPS, higher inner product is better. To sort ascending by distance,
      // store -inner_product so the maximum inner product comes first.
      return -inner_product_scalar(row, query, dim);
    case MetricType::L2:
    default:
      return l2_squared_scalar(row, query, dim);
  }
}

} // anonymous namespace

std::vector<SearchResult> linear_scan(const std::vector<float> &dataset,
                                      const std::vector<float> &query,
                                      int top_k,
                                      MetricType metric) {
  if (query.empty() || dataset.empty()) {
    return {};
  }

  const size_t dim = query.size();
  const size_t N = dataset.size() / dim;
  if (N == 0) {
    return {};
  }

  const size_t safe_top_k = std::min(static_cast<size_t>(std::max(0, top_k)), N);
  if (safe_top_k == 0) {
    return {};
  }

  std::vector<SearchResult> results;
  results.reserve(N);

  for (size_t i = 0; i < N; ++i) {
    const float *row = &dataset[i * dim];
    float dist = compute_distance(metric, row, query.data(), dim);
    results.push_back({static_cast<int>(i), dist});
  }

  std::partial_sort(results.begin(), results.begin() + safe_top_k, results.end(),
                    [](const SearchResult &a, const SearchResult &b) {
                      return a.distance < b.distance;
                    });
  results.resize(safe_top_k);
  return results;
}

std::vector<SearchResult> linear_scan(const std::vector<float> &dataset,
                                      const std::vector<float> &query,
                                      int top_k,
                                      const std::string &method) {
  return linear_scan(dataset, query, top_k, parse_metric(method));
}

std::vector<SearchResult> linear_scan(const FloatDataset &dataset,
                                      const float *query,
                                      size_t top_k,
                                      MetricType metric) {
  if (dataset.empty() || query == nullptr || dataset.dim == 0) {
    return {};
  }

  const size_t N = dataset.num_vectors;
  const size_t dim = dataset.dim;
  const size_t safe_top_k = std::min(top_k, N);
  if (safe_top_k == 0) {
    return {};
  }

  std::vector<SearchResult> results;
  results.reserve(N);

  for (size_t i = 0; i < N; ++i) {
    const float *row = dataset.get(i);
    float dist = compute_distance(metric, row, query, dim);
    results.push_back({static_cast<int>(i), dist});
  }

  std::partial_sort(results.begin(), results.begin() + safe_top_k, results.end(),
                    [](const SearchResult &a, const SearchResult &b) {
                      return a.distance < b.distance;
                    });
  results.resize(safe_top_k);
  return results;
}

std::vector<SearchResult> linear_scan(const FloatDataset &dataset,
                                      const float *query,
                                      size_t top_k,
                                      const std::string &method) {
  return linear_scan(dataset, query, top_k, parse_metric(method));
}

std::vector<SearchResult> linear_scan_with_prefetch(
    const FloatDataset &dataset,
    const float *query,
    size_t top_k,
    MetricType metric,
    size_t prefetch_ahead) {
  if (dataset.empty() || query == nullptr || dataset.dim == 0) {
    return {};
  }

  const size_t N = dataset.num_vectors;
  const size_t dim = dataset.dim;
  const size_t safe_top_k = std::min(top_k, N);
  if (safe_top_k == 0) {
    return {};
  }

  std::vector<SearchResult> results;
  results.reserve(N);

  const size_t bytes_per_vector = dim * sizeof(float);

  for (size_t i = 0; i < N; ++i) {
    // Software prefetch lookahead vectors into L1 cache
    if (prefetch_ahead > 0 && i + prefetch_ahead < N) {
      const char *prefetch_ptr = reinterpret_cast<const char *>(dataset.get(i + prefetch_ahead));
      for (size_t line = 0; line < bytes_per_vector; line += 64) {
        _mm_prefetch(prefetch_ptr + line, _MM_HINT_T0);
      }
    }

    const float *row = dataset.get(i);
    float dist = compute_distance(metric, row, query, dim);
    results.push_back({static_cast<int>(i), dist});
  }

  std::partial_sort(results.begin(), results.begin() + safe_top_k, results.end(),
                    [](const SearchResult &a, const SearchResult &b) {
                      return a.distance < b.distance;
                    });
  results.resize(safe_top_k);
  return results;
}

std::vector<SearchResult> linear_scan_with_prefetch(
    const FloatDataset &dataset,
    const float *query,
    size_t top_k,
    const std::string &method,
    size_t prefetch_ahead) {
  return linear_scan_with_prefetch(dataset, query, top_k, parse_metric(method), prefetch_ahead);
}

std::vector<TopKQueryResult> batch_linear_scan_tiled(
    const float *queries, size_t num_queries,
    const float *base, size_t num_base,
    size_t dim, size_t k,
    size_t tile_size) {
  if (num_queries == 0 || num_base == 0 || dim == 0 || k == 0 || queries == nullptr || base == nullptr) {
    return std::vector<TopKQueryResult>(num_queries);
  }

  const size_t safe_k = std::min(k, num_base);
  std::vector<std::priority_queue<std::pair<float, int32_t>>> heaps(num_queries);

  // Outer loop over database tiles (streamed once from DRAM into L2/L3 cache)
  for (size_t t = 0; t < num_base; t += tile_size) {
    const size_t cur_tile_size = std::min(tile_size, num_base - t);
    const float *tile_data = base + t * dim;

    // Inner loop over queries (reusing the tile resident in cache)
    for (size_t q = 0; q < num_queries; ++q) {
      const float *q_vec = queries + q * dim;
      auto &heap = heaps[q];

      for (size_t i = 0; i < cur_tile_size; ++i) {
        const int32_t global_idx = static_cast<int32_t>(t + i);
        const float *b_vec = tile_data + i * dim;
        const float dist = l2_squared_avx2_unroll4(q_vec, b_vec, dim);

        if (heap.size() < safe_k) {
          heap.emplace(dist, global_idx);
        } else if (dist < heap.top().first) {
          heap.pop();
          heap.emplace(dist, global_idx);
        }
      }
    }
  }

  std::vector<TopKQueryResult> results(num_queries);
  for (size_t q = 0; q < num_queries; ++q) {
    auto &heap = heaps[q];
    results[q].indices.resize(heap.size());
    results[q].distances.resize(heap.size());
    for (int i = static_cast<int>(heap.size()) - 1; i >= 0; --i) {
      results[q].distances[i] = heap.top().first;
      results[q].indices[i] = heap.top().second;
      heap.pop();
    }
  }
  return results;
}

std::vector<TopKQueryResult> batch_linear_scan_tiled(
    const FloatDataset &dataset,
    const float *queries, size_t num_queries,
    size_t top_k,
    size_t tile_size) {
  return batch_linear_scan_tiled(queries, num_queries, dataset.data.data(),
                                 dataset.num_vectors, dataset.dim, top_k, tile_size);
}

} // namespace secan

