#include "secan/search/search.h"
#include "test_utils.h"

void test_linear_scan_finds_exact_match() {
  // dataset: 3 vectors of dim 2
  std::vector<float> dataset = {1.0f, 1.0f, 5.0f, 5.0f, 9.0f, 9.0f};
  std::vector<float> query = {5.0f, 5.0f};

  auto results = linear_scan(dataset, query, 1, "l2");
  CHECK(results[0].index == 1); // exact match at index 1
  CHECK(results[0].distance == 0.0f);
}

int main() {
  test_linear_scan_finds_exact_match();
  return report_results("linear_scan_tests");
}
