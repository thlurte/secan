#ifndef SECAN_SEARCH_SEARCH_H
#define SECAN_SEARCH_SEARCH_H

#include "secan/utils/io.h"
#include "secan/utils/utils.h"
#include <string>
#include <vector>

namespace secan {

std::vector<SearchResult> linear_scan(const std::vector<float> &dataset,
                                      const std::vector<float> &query,
                                      int top_k,
                                      const std::string &method = "cosine");

std::vector<SearchResult> linear_scan(const FloatDataset &dataset,
                                      const float *query,
                                      size_t top_k,
                                      const std::string &method = "l2");

} // namespace secan

// Backward compatibility in global namespace
using secan::linear_scan;

#endif // SECAN_SEARCH_SEARCH_H
