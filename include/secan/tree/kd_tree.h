#ifndef SECAN_TREE_KD_TREE_H
#define SECAN_TREE_KD_TREE_H

#include "secan/search/distance.h"
#include "secan/utils/io.h"
#include "secan/utils/utils.h"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

namespace secan {

struct KdNode {
  int split_dim{-1};       // Dimension used for split (-1 if leaf)
  float split_val{0.0f};   // Value used for split
  int left{-1};            // Index of left child in tree node array
  int right{-1};           // Index of right child in tree node array
  std::vector<int32_t> indices; // Vector indices if leaf node

  [[nodiscard]] bool is_leaf() const noexcept { return split_dim < 0; }
};

struct KdTreeParams {
  size_t num_trees{4};           // Number of randomized KD-trees in ensemble
  size_t leaf_max_size{32};      // Max points in a leaf node
  size_t top_variance_dims{5};   // Top variance dimensions to randomly choose from
  unsigned int seed{42};         // Random seed for reproducibility
};

struct KdSearchParams {
  size_t max_checks{512};        // Search budget: maximum number of leaf vectors to evaluate
  MetricType metric{MetricType::L2};
};

class RandomizedKdTree {
public:
  RandomizedKdTree() = default;
  explicit RandomizedKdTree(KdTreeParams params);

  /// Build index over a dataset
  void build(const float *data, size_t num_vectors, size_t dim);
  void build(const FloatDataset &dataset);

  /// Search top-k approximate nearest neighbors for a single query using Best-Bin-First (BBF)
  [[nodiscard]] std::vector<SearchResult> search(
      const float *query,
      size_t top_k,
      const KdSearchParams &search_params = {}) const;

  [[nodiscard]] std::vector<SearchResult> search(
      const std::vector<float> &query,
      size_t top_k,
      const KdSearchParams &search_params = {}) const;

  /// Batch search over multiple queries
  [[nodiscard]] std::vector<std::vector<SearchResult>> batch_search(
      const float *queries,
      size_t num_queries,
      size_t top_k,
      const KdSearchParams &search_params = {}) const;

  [[nodiscard]] size_t size() const noexcept { return num_vectors_; }
  [[nodiscard]] size_t dim() const noexcept { return dim_; }
  [[nodiscard]] size_t num_trees() const noexcept { return trees_.size(); }

private:
  struct SingleTree {
    std::vector<KdNode> nodes;
    int root{-1};
  };

  int build_node(SingleTree &tree,
                 std::vector<int32_t> &indices,
                 size_t start,
                 size_t end,
                 std::mt19937 &rng);

  KdTreeParams params_;
  const float *data_{nullptr};
  size_t num_vectors_{0};
  size_t dim_{0};
  std::vector<SingleTree> trees_;
};

} // namespace secan

#endif // SECAN_TREE_KD_TREE_H
