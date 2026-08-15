#ifndef SECAN_SEARCH_SEARCH_H
#define SECAN_SEARCH_SEARCH_H

#include "secan/utils/utils.h"
#include <string>
#include <vector>

std::vector<SearchResult> linear_scan(const std::vector<float> &dataset,
                                      const std::vector<float> &query,
                                      int top_k, std::string method = "cosine");

#endif
