#include "secan/search/distance.h"
#include "secan/utils/utils.h"
#include <algorithm>
#include <string>
#include <vector>

std::vector<SearchResult> linear_scan(const std::vector<float> &dataset,
                                      const std::vector<float> &query,
                                      int top_k,
                                      std::string method = "cosine") {

  int dim = query.size();
  int N = dataset.size() / dim;
  std::vector<SearchResult> results;
  results.reserve(N);
  for (size_t i = 0; static_cast<size_t>(i) < N; i++) {
    const float *row = &dataset[i * dim];
    if (method == "cosine") {
      float dist = cosine_distance(row, query.data(), dim);
      results.push_back({static_cast<int>(i), dist});

    } else {
      float dist = l2_squared(row, query.data(), dim);
      results.push_back({static_cast<int>(i), dist});
    }
  }
  std::partial_sort(results.begin(), results.begin() + top_k, results.end(),
                    [](const SearchResult &a, const SearchResult &b) {
                      return a.distance < b.distance;
                    });
  results.resize(top_k);
  return results;
}
