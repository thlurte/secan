#include "secan/utils/normalize.h"
#include <algorithm>
#include <cmath>
#include <immintrin.h>

namespace secan {

namespace {

inline float horizontal_sum_avx2(__m256 v) noexcept {
  __m128 lo = _mm256_castps256_ps128(v);
  __m128 hi = _mm256_extractf128_ps(v, 1);
  __m128 sum128 = _mm_add_ps(lo, hi);
  sum128 = _mm_hadd_ps(sum128, sum128);
  sum128 = _mm_hadd_ps(sum128, sum128);
  return _mm_cvtss_f32(sum128);
}

} // anonymous namespace

void normalize_vector_avx2(float *vec, size_t dim) noexcept {
  if (vec == nullptr || dim == 0) {
    return;
  }

  __m256 sum = _mm256_setzero_ps();
  size_t i = 0;
  for (; i + 8 <= dim; i += 8) {
    __m256 v = _mm256_loadu_ps(vec + i);
    sum = _mm256_fmadd_ps(v, v, sum);
  }

  float norm_sq = horizontal_sum_avx2(sum);
  for (; i < dim; ++i) {
    norm_sq += vec[i] * vec[i];
  }

  if (norm_sq > 1e-12f) {
    float inv_norm = 1.0f / std::sqrt(norm_sq);
    __m256 vinv = _mm256_set1_ps(inv_norm);

    i = 0;
    for (; i + 8 <= dim; i += 8) {
      __m256 v = _mm256_loadu_ps(vec + i);
      _mm256_storeu_ps(vec + i, _mm256_mul_ps(v, vinv));
    }
    for (; i < dim; ++i) {
      vec[i] *= inv_norm;
    }
  }
}

void normalize_dataset_avx2(float *data, size_t num_vectors, size_t dim) noexcept {
  if (data == nullptr || num_vectors == 0 || dim == 0) {
    return;
  }

  for (size_t n = 0; n < num_vectors; ++n) {
    float *vec = data + n * dim;
    normalize_vector_avx2(vec, dim);
  }
}

void normalize_dataset_stream(float *data, size_t num_vectors, size_t dim) noexcept {
  if (data == nullptr || num_vectors == 0 || dim == 0) {
    return;
  }

  for (size_t n = 0; n < num_vectors; ++n) {
    float *vec = data + n * dim;

    // 1. Compute norm squared
    __m256 sum = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 8 <= dim; i += 8) {
      __m256 v = _mm256_loadu_ps(vec + i);
      sum = _mm256_fmadd_ps(v, v, sum);
    }

    float norm_sq = horizontal_sum_avx2(sum);
    for (; i < dim; ++i) {
      norm_sq += vec[i] * vec[i];
    }

    if (norm_sq > 1e-12f) {
      float inv_norm = 1.0f / std::sqrt(norm_sq);
      __m256 vinv = _mm256_set1_ps(inv_norm);

      i = 0;
      // If the address is 32-byte aligned, use _mm256_stream_ps; otherwise standard store
      const bool is_aligned = (reinterpret_cast<uintptr_t>(vec) % 32 == 0);

      if (is_aligned) {
        for (; i + 8 <= dim; i += 8) {
          __m256 v = _mm256_loadu_ps(vec + i);
          _mm256_stream_ps(vec + i, _mm256_mul_ps(v, vinv));
        }
      } else {
        for (; i + 8 <= dim; i += 8) {
          __m256 v = _mm256_loadu_ps(vec + i);
          _mm256_storeu_ps(vec + i, _mm256_mul_ps(v, vinv));
        }
      }
      for (; i < dim; ++i) {
        vec[i] *= inv_norm;
      }
    }
  }

  // Memory fence for non-temporal stores
  _mm_sfence();
}

void normalize_dataset(FloatDataset &dataset, bool use_streaming) {
  if (dataset.empty() || dataset.dim == 0) {
    return;
  }
  if (use_streaming) {
    normalize_dataset_stream(dataset.data.data(), dataset.num_vectors, dataset.dim);
  } else {
    normalize_dataset_avx2(dataset.data.data(), dataset.num_vectors, dataset.dim);
  }
}

} // namespace secan
