#ifndef SECAN_UTILS_NORMALIZE_H
#define SECAN_UTILS_NORMALIZE_H

#include "secan/utils/io.h"
#include <cstddef>

namespace secan {

/// Normalizes a single vector in-place using AVX2.
void normalize_vector_avx2(float *vec, size_t dim) noexcept;

/// Normalizes a dataset of vectors in-place using standard AVX2 stores.
void normalize_dataset_avx2(float *data, size_t num_vectors, size_t dim) noexcept;

/// Normalizes a dataset of vectors in-place using non-temporal streaming stores (_mm256_stream_ps).
/// Useful for massive datasets to prevent polluting the L1/L2/L3 CPU cache hierarchy.
void normalize_dataset_stream(float *data, size_t num_vectors, size_t dim) noexcept;

/// Normalizes a FloatDataset in-place.
void normalize_dataset(FloatDataset &dataset, bool use_streaming = false);

} // namespace secan

#endif // SECAN_UTILS_NORMALIZE_H
