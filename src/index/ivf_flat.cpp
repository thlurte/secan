#include "secan/index/ivf_flat.h"
#include "secan/search/distance.h"
#include "secan/utils/utils.h"
#include <algorithm>
#include <cstring>
#include <numeric>
#include <queue>
#include <random>
#include <stdexcept>

namespace secan {

void InvertedList::reserve(size_t n_vecs, size_t dim) {
  ids.reserve(n_vecs);
  data.reserve(n_vecs * dim);
}

void InvertedList::add(int32_t id, const float *vec, size_t dim) {
  ids.push_back(id);
  data.insert(data.end(), vec, vec + dim);
}

void InvertedList::clear() noexcept {
  ids.clear();
  data.clear();
}

IvfFlatIndex::IvfFlatIndex(size_t dim, size_t nlist, MetricType metric)
    : dim_(dim), nlist_(nlist), metric_(metric), is_trained_(false),
      centroids_(nlist * dim, 0.0f), lists_(nlist) {}
size_t IvfFlatIndex::total_vectors() const noexcept {
  size_t total = 0;
  for (const auto &list : lists_) {
    total += list.size();
  }
  return total;
}

void IvfFlatIndex::train(size_t n, const float *data, size_t max_iters) {
  if (n < nlist_) {
    throw std::invalid_argument("Training data size must be >= nlist");
  }
  // array with the capacity of n
  std::vector<size_t> indices(n);
  // populate arr with range numbers incrementing upto n
  std::iota(indices.begin(), indices.end(), 0);

  std::mt19937 rng(42);
  // shuffle the elements inside the arr
  std::shuffle(indices.begin(), indices.end(), rng);

  // loop up to nlist
  for (size_t k = 0; k < nlist_; ++k) {
    // of the shuffled elements obtain kth element
    size_t idx = indices[k];
    // copy ith element from data into centroids kth location
    std::memcpy(centroids_.data() + (k * dim_), data + (idx * dim_),
                dim_ * sizeof(float));
  }

  for (size_t iter = 0; iter < max_iters; ++iter) {

    std::vector<float> accumulators(nlist_ * dim_, 0.0f);
    std::vector<size_t> counts(nlist_, 0);

    // b. For each vector x in data
    for (size_t i = 0; i < n; ++i) {
      const float *x = data + (i * dim_);

      // Find nearest centroid k*
      size_t best_k = 0;
      float min_dist = std::numeric_limits<float>::max();

      for (size_t k = 0; k < nlist_; ++k) {
        const float *c = centroids_.data() + (k * dim_);

        float dist = 0.0f;
        if (metric_ == MetricType::L2) {
          dist = l2_squared_scalar(x, c, dim_);
        } else if (metric_ == MetricType::Cosine) {
          dist = cosine_distance_fast_scalar(x, c, dim_);
        }

        if (dist < min_dist) {
          min_dist = dist;
          best_k = k;
        }
      }
      // accumulators[k*] += x
      for (size_t d = 0; d < dim_; ++d) {
        accumulators[(best_k * dim_) + d] += x[d];
      }

      // counts[k*]++
      counts[best_k]++;
    }
    // c. centroids[k] = accumulators[k] / counts[k]
    for (size_t k = 0; k < nlist_; ++k) {
      // CRITICAL: Prevent division by zero (NaN) if a cluster becomes empty
      if (counts[k] > 0) {
        float norm_sq = 0.0f;

        for (size_t d = 0; d < dim_; ++d) {
          float val = accumulators[(k * dim_) + d] / counts[k];
          centroids_[(k * dim_) + d] = val;
          norm_sq += val * val;
        }

        if (metric_ == MetricType::Cosine && norm_sq > 1e-12f) {
          // You can use your fast_rsqrt here!
          float inv_norm = fast_rsqrt(norm_sq);

          for (size_t d = 0; d < dim_; ++d) {
            centroids_[(k * dim_) + d] *=
                inv_norm; // Stretch it back to length 1.0
          }
        }
      }
    }
  }

  is_trained_ = true;
}
} // namespace secan
