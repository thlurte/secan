#ifndef SECAN_SEARCH_RERANKER_H
#define SECAN_SEARCH_RERANKER_H

#include "secan/quantization/scalar_quantizer.h"
#include "secan/search/distance.h"
#include "secan/utils/utils.h"
#include <vector>
#include <cstddef>
#include <cstdint>

namespace secan {

/**
 * @brief Two-Stage Search & Re-ranking Engine.
 * 
 * Stage 1: Ultra-fast low-bit (SQ4/SQ8) coarse scan over entire dataset to prune candidates to top-M.
 * Stage 2: Exact FP32 distance re-ranking over top-M filtered candidates to restore high recall (>99.5%).
 */
class TwoStageReranker {
public:
    TwoStageReranker(
        const ScalarQuantizer4& sq4,
        const uint8_t* packed_sq4_data,
        const float* raw_fp32_data,
        size_t num_vectors,
        size_t dim
    );

    /**
     * @brief Performs two-stage search: coarse SQ4 scan (top-M) -> exact FP32 re-ranking (top-K).
     * @param query Pointer to float32 query vector (dim).
     * @param top_k Number of final nearest neighbors to return.
     * @param coarse_m Number of coarse candidates to select in Stage 1 (default 100).
     */
    [[nodiscard]] std::vector<SearchResult> search(
        const float* query,
        size_t top_k,
        size_t coarse_m = 100
    ) const;

private:
    const ScalarQuantizer4& sq4_;
    const uint8_t* packed_sq4_data_{nullptr};
    const float* raw_fp32_data_{nullptr};
    size_t num_vectors_{0};
    size_t dim_{0};
};

} // namespace secan

#endif // SECAN_SEARCH_RERANKER_H
