#include "secan/utils/utils.h"
#include <random>

std::pair<std::vector<float>, std::vector<int>>
generate(float min, float max, std::vector<int> shape) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(min, max);

  int length = 1;
  for (const auto &dim : shape) {
    length *= dim;
  }
  std::vector<float> vec;
  vec.reserve(length);

  for (size_t i = 0; i < static_cast<size_t>(length); i++) {
    vec.push_back(dist(gen));
  }
  return {vec, shape};
}
