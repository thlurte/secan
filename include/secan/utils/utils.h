#ifndef SECAN_UTILS_UTILS_H
#define SECAN_UTILS_UTILS_H

#include <utility>
#include <vector>

struct SearchResult {
  int index;
  float distance;
};

std::pair<std::vector<float>, std::vector<int>>
generate(float min, float max, std::vector<int> shape);

#endif
