#ifndef SECAN_UTILS_METRICS_H
#define SECAN_UTILS_METRICS_H

#include <cstddef>
#include <cstdint>
#include <vector>

namespace secan {

/// Compute Recall@K for a single query.
/// k is the evaluation cutoff.
/// retrieved: vector of document IDs returned by search, sorted by ranking score/distance.
/// ground_truth: vector of true nearest neighbor IDs.
double recall_at_k(const std::vector<int32_t> &retrieved,
                   const std::vector<int32_t> &ground_truth,
                   size_t k);

/// Compute Reciprocal Rank (RR) for a single query against ground truth set.
/// Returns 1.0 / rank (1-based) of the first relevant document in retrieved.
double reciprocal_rank(const std::vector<int32_t> &retrieved,
                       const std::vector<int32_t> &ground_truth);

/// Compute Reciprocal Rank for a specific relevant document ID.
double reciprocal_rank(const std::vector<int32_t> &retrieved,
                       int32_t relevant_id);

/// Compute Normalized Discounted Cumulative Gain (NDCG@K) with logarithmic decay.
double ndcg_at_k(const std::vector<int32_t> &retrieved,
                 const std::vector<int32_t> &ground_truth,
                 size_t k);

/// Compute Average Precision (AP@K) for a single query.
double average_precision_at_k(const std::vector<int32_t> &retrieved,
                              const std::vector<int32_t> &ground_truth,
                              size_t k);

/// Compute Mean Recall@K across multiple queries.
double mean_recall_at_k(const std::vector<std::vector<int32_t>> &all_retrieved,
                        const std::vector<std::vector<int32_t>> &all_ground_truth,
                        size_t k);

/// Compute Mean Reciprocal Rank (MRR) across multiple queries.
double mean_reciprocal_rank(const std::vector<std::vector<int32_t>> &all_retrieved,
                            const std::vector<std::vector<int32_t>> &all_ground_truth);

/// Compute Mean NDCG@K across multiple queries.
double mean_ndcg_at_k(const std::vector<std::vector<int32_t>> &all_retrieved,
                      const std::vector<std::vector<int32_t>> &all_ground_truth,
                      size_t k);

/// Compute Mean Average Precision (MAP@K) across multiple queries.
double mean_average_precision(const std::vector<std::vector<int32_t>> &all_retrieved,
                              const std::vector<std::vector<int32_t>> &all_ground_truth,
                              size_t k);

} // namespace secan

#endif // SECAN_UTILS_METRICS_H
