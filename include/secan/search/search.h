#ifndef SECAN_SEARCH_SEARCH_H
#define SECAN_SEARCH_SEARCH_H

#include "secan/search/distance.h"
#include "secan/utils/io.h"
#include "secan/utils/utils.h"
#include <string>
#include <vector>

namespace secan {

std::vector<SearchResult> linear_scan(const std::vector<float> &dataset,
                                      const std::vector<float> &query,
                                      int top_k,
                                      const std::string &method = "cosine");

std::vector<SearchResult> linear_scan(const std::vector<float> &dataset,
                                      const std::vector<float> &query,
                                      int top_k,
                                      MetricType metric);

std::vector<SearchResult> linear_scan(const FloatDataset &dataset,
                                      const float *query,
                                      size_t top_k,
                                      const std::string &method = "l2");

std::vector<SearchResult> linear_scan(const FloatDataset &dataset,
                                      const float *query,
                                      size_t top_k,
                                      MetricType metric);

struct TopKQueryResult {
  std::vector<int32_t> indices;
  std::vector<float> distances;
};

// Batched query scan over database with L2/L3 cache tiling
std::vector<TopKQueryResult> batch_linear_scan_tiled(
    const float *queries, size_t num_queries,
    const float *base, size_t num_base,
    size_t dim, size_t k,
    size_t tile_size = 2048);

std::vector<TopKQueryResult> batch_linear_scan_tiled(
    const FloatDataset &dataset,
    const float *queries, size_t num_queries,
    size_t top_k,
    size_t tile_size = 2048);

} // namespace secan

// Backward compatibility in global namespace
using secan::linear_scan;
using secan::batch_linear_scan_tiled;
using secan::TopKQueryResult;

#endif // SECAN_SEARCH_SEARCH_H
