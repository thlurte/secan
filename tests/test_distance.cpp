#include "secan/search/distance.h"
#include "test_utils.h"
#include <cmath>

void test_l2_squared() {
  float a[] = {1.0f, 2.0f, 3.0f};
  float b[] = {4.0f, 5.0f, 6.0f};
  CHECK(l2_squared(a, b, 3) == 27.0f);

  float same[] = {1.0f, 2.0f, 3.0f};
  CHECK(l2_squared(same, same, 3) == 0.0f);
}

void test_cosine_distance() {
  float a[] = {1.0f, 2.0f, 3.0f};
  CHECK_NEAR(cosine_distance(a, a, 3), 0.0f, 1e-5f);
}

int main() {
  test_l2_squared();
  test_cosine_distance();
  return report_results("distance_tests");
}
