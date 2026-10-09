#include "secan/search/reranker.h"
#include "secan/search/distance_avx2.h"
#include <algorithm>
#include <stdexcept>
#include <vector>

namespace secan {

TwoStageReranker::TwoStageReranker(
    const ScalarQuantizer4& sq4,
    const uint8_t* packed_sq4_data,
    const float* raw_fp32_data,
    size_t num_vectors,
    size_t dim
) : sq4_(sq4),
    packed_sq4_data_(packed_sq4_data),
    raw_fp32_data_(raw_fp32_data),
    num_vectors_(num_vectors),
    dim_(dim) {
    if (packed_sq4_data == nullptr || raw_fp32_data == nullptr || num_vectors == 0 || dim == 0) {
        throw std::invalid_argument("TwoStageReranker: Invalid data pointers or empty dataset");
    }
}

std::vector<SearchResult> TwoStageReranker::search(
    const float* query,
    size_t top_k,
    size_t coarse_m
) const {
    if (query == nullptr || top_k == 0) {
        return {};
    }

    size_t m = std::clamp(std::max(top_k, coarse_m), size_t(1), num_vectors_);
    size_t k = std::min(top_k, m);

    // Stage 1: Coarse SQ4 scan across all N vectors
    std::vector<SearchResult> candidates(num_vectors_);
    size_t bytes_per_vec = sq4_.bytes_per_vector();

    for (size_t i = 0; i < num_vectors_; ++i) {
        float approx_dist = sq4_.compute_distance_l2(query, packed_sq4_data_ + i * bytes_per_vec);
        candidates[i] = SearchResult{static_cast<int>(i), approx_dist};
    }

    // Partial sort to find top-M coarse candidates in O(N log M)
    std::partial_sort(
        candidates.begin(),
        candidates.begin() + static_cast<std::ptrdiff_t>(m),
        candidates.end(),
        [](const SearchResult& a, const SearchResult& b) {
            return a.distance < b.distance;
        }
    );

    // Stage 2: Exact FP32 re-ranking on the filtered top-M candidates
    std::vector<SearchResult> refined(m);
    for (size_t i = 0; i < m; ++i) {
        size_t orig_id = static_cast<size_t>(candidates[i].index);
        const float* raw_vec = raw_fp32_data_ + orig_id * dim_;
        
        // Exact L2 distance via AVX2 unrolled accumulator
        float exact_dist = l2_squared_avx2_unroll4(query, raw_vec, dim_);
        refined[i] = SearchResult{static_cast<int>(orig_id), exact_dist};
    }

    // Sort top-K refined results
    std::partial_sort(
        refined.begin(),
        refined.begin() + static_cast<std::ptrdiff_t>(k),
        refined.end(),
        [](const SearchResult& a, const SearchResult& b) {
            return a.distance < b.distance;
        }
    );

    refined.resize(k);
    return refined;
}

} // namespace secan
