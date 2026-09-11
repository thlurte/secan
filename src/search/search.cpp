#include "secan/search/search.h"
#include "secan/search/distance.h"
#include <algorithm>
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

} // namespace secan
