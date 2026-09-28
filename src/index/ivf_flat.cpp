#include "secan/index/ivf_flat.h"
#include "secan/search/distance.h"
#include "secan/utils/utils.h"
#include <algorithm>
#include <cstring>
#include <queue>

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

} // namespace secan
