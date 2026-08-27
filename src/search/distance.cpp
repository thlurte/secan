#include <cmath>
#include <immintrin.h>

float l2_squared(const float *row, const float *query, int dim) {
  float diff = 0.0f;
  for (size_t i = 0; i < static_cast<size_t>(dim); i++) {
    diff += (row[i] - query[i]) * (row[i] - query[i]);
  }
  return diff;
}

float cosine_distance(const float *row, const float *query, int dim) {
  float dot_prod = 0.0f;
  float norm_query = 0.0f;
  float norm_row = 0.0f;
  for (size_t i = 0; i < dim; i++) {
    dot_prod = dot_prod + (row[i] * query[i]);
    norm_query += query[i] * query[i];
    norm_row += row[i] * row[i];
  }
  float similarity =
      dot_prod / std::sqrt(norm_query * norm_row); // overflow risk is possible
  return 1.0f - similarity;
}

float l2_squared_avx2(const float *row, const float *query, int dim) {
  __m256 acc = _mm256_setzero_ps();
}
