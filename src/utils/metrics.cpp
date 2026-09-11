#include "secan/utils/metrics.h"

#include <algorithm>
#include <cmath>
#include <unordered_set>

namespace secan {

double recall_at_k(const std::vector<int32_t> &retrieved,
                   const std::vector<int32_t> &ground_truth,
                   size_t k) {
  if (k == 0 || ground_truth.empty() || retrieved.empty()) {
    return 0.0;
  }

  const size_t eval_gt = std::min(k, ground_truth.size());
  const size_t eval_retrieved = std::min(k, retrieved.size());

  std::unordered_set<int32_t> gt_set(ground_truth.begin(),
                                     ground_truth.begin() + eval_gt);

  size_t hits = 0;
  for (size_t i = 0; i < eval_retrieved; ++i) {
    if (gt_set.contains(retrieved[i])) {
      hits++;
    }
  }

  return static_cast<double>(hits) / static_cast<double>(k);
}

double reciprocal_rank(const std::vector<int32_t> &retrieved,
                       const std::vector<int32_t> &ground_truth) {
  if (retrieved.empty() || ground_truth.empty()) {
    return 0.0;
  }

  std::unordered_set<int32_t> gt_set(ground_truth.begin(), ground_truth.end());
  for (size_t i = 0; i < retrieved.size(); ++i) {
    if (gt_set.contains(retrieved[i])) {
      return 1.0 / static_cast<double>(i + 1);
    }
  }

  return 0.0;
}

double reciprocal_rank(const std::vector<int32_t> &retrieved,
                       int32_t relevant_id) {
  for (size_t i = 0; i < retrieved.size(); ++i) {
    if (retrieved[i] == relevant_id) {
      return 1.0 / static_cast<double>(i + 1);
    }
  }
  return 0.0;
}

double ndcg_at_k(const std::vector<int32_t> &retrieved,
                 const std::vector<int32_t> &ground_truth,
                 size_t k) {
  if (k == 0 || ground_truth.empty() || retrieved.empty()) {
    return 0.0;
  }

  const size_t eval_gt = std::min(k, ground_truth.size());
  const size_t eval_retrieved = std::min(k, retrieved.size());

  std::unordered_set<int32_t> gt_set(ground_truth.begin(),
                                     ground_truth.begin() + eval_gt);

  double dcg = 0.0;
  for (size_t i = 0; i < eval_retrieved; ++i) {
    if (gt_set.contains(retrieved[i])) {
      // 0-indexed rank i -> position i+1 -> log2((i+1) + 1) = log2(i + 2)
      dcg += 1.0 / std::log2(static_cast<double>(i + 2));
    }
  }

  double idcg = 0.0;
  for (size_t i = 0; i < eval_gt; ++i) {
    idcg += 1.0 / std::log2(static_cast<double>(i + 2));
  }

  return (idcg > 0.0) ? (dcg / idcg) : 0.0;
}

double average_precision_at_k(const std::vector<int32_t> &retrieved,
                              const std::vector<int32_t> &ground_truth,
                              size_t k) {
  if (k == 0 || ground_truth.empty() || retrieved.empty()) {
    return 0.0;
  }

  const size_t eval_gt = std::min(k, ground_truth.size());
  const size_t eval_retrieved = std::min(k, retrieved.size());

  std::unordered_set<int32_t> gt_set(ground_truth.begin(),
                                     ground_truth.begin() + eval_gt);

  size_t hits = 0;
  double precision_sum = 0.0;

  for (size_t i = 0; i < eval_retrieved; ++i) {
    if (gt_set.contains(retrieved[i])) {
      hits++;
      double precision_at_i = static_cast<double>(hits) / static_cast<double>(i + 1);
      precision_sum += precision_at_i;
    }
  }

  return (eval_gt > 0) ? (precision_sum / static_cast<double>(eval_gt)) : 0.0;
}

double mean_recall_at_k(const std::vector<std::vector<int32_t>> &all_retrieved,
                        const std::vector<std::vector<int32_t>> &all_ground_truth,
                        size_t k) {
  if (all_retrieved.empty() || all_retrieved.size() != all_ground_truth.size()) {
    return 0.0;
  }

  double total_recall = 0.0;
  for (size_t q = 0; q < all_retrieved.size(); ++q) {
    total_recall += recall_at_k(all_retrieved[q], all_ground_truth[q], k);
  }

  return total_recall / static_cast<double>(all_retrieved.size());
}

double mean_reciprocal_rank(const std::vector<std::vector<int32_t>> &all_retrieved,
                            const std::vector<std::vector<int32_t>> &all_ground_truth) {
  if (all_retrieved.empty() || all_retrieved.size() != all_ground_truth.size()) {
    return 0.0;
  }

  double total_rr = 0.0;
  for (size_t q = 0; q < all_retrieved.size(); ++q) {
    total_rr += reciprocal_rank(all_retrieved[q], all_ground_truth[q]);
  }

  return total_rr / static_cast<double>(all_retrieved.size());
}

double mean_ndcg_at_k(const std::vector<std::vector<int32_t>> &all_retrieved,
                      const std::vector<std::vector<int32_t>> &all_ground_truth,
                      size_t k) {
  if (all_retrieved.empty() || all_retrieved.size() != all_ground_truth.size()) {
    return 0.0;
  }

  double total_ndcg = 0.0;
  for (size_t q = 0; q < all_retrieved.size(); ++q) {
    total_ndcg += ndcg_at_k(all_retrieved[q], all_ground_truth[q], k);
  }

  return total_ndcg / static_cast<double>(all_retrieved.size());
}

double mean_average_precision(const std::vector<std::vector<int32_t>> &all_retrieved,
                              const std::vector<std::vector<int32_t>> &all_ground_truth,
                              size_t k) {
  if (all_retrieved.empty() || all_retrieved.size() != all_ground_truth.size()) {
    return 0.0;
  }

  double total_ap = 0.0;
  for (size_t q = 0; q < all_retrieved.size(); ++q) {
    total_ap += average_precision_at_k(all_retrieved[q], all_ground_truth[q], k);
  }

  return total_ap / static_cast<double>(all_retrieved.size());
}

} // namespace secan
