#include "secan/tree/kd_tree.h"
#include "secan/search/distance.h"
#include "secan/search/distance_avx2.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <queue>
#include <stdexcept>

namespace secan {

namespace {

inline float compute_vector_distance(MetricType metric, const float *a, const float *b, size_t dim) {
  switch (metric) {
    case MetricType::Cosine:
      return cosine_distance_avx2(a, b, dim);
    case MetricType::IP:
      // For ascending priority / distance sort, use -dot product
      return -ip_avx2_unroll4(a, b, dim);
    case MetricType::L2:
    default:
      return l2_squared_avx2_unroll4(a, b, dim);
  }
}

} // anonymous namespace

RandomizedKdTree::RandomizedKdTree(KdTreeParams params)
    : params_(params) {}

void RandomizedKdTree::build(const FloatDataset &dataset) {
  build(dataset.data.data(), dataset.num_vectors, dataset.dim);
}

void RandomizedKdTree::build(const float *data, size_t num_vectors, size_t dim) {
  if (data == nullptr || num_vectors == 0 || dim == 0) {
    trees_.clear();
    data_ = nullptr;
    num_vectors_ = 0;
    dim_ = 0;
    return;
  }

  data_ = data;
  num_vectors_ = num_vectors;
  dim_ = dim;
  trees_.resize(params_.num_trees);

  std::mt19937 rng(params_.seed);

  for (size_t t = 0; t < params_.num_trees; ++t) {
    SingleTree &tree = trees_[t];
    tree.nodes.clear();
    tree.nodes.reserve(2 * (num_vectors / std::max<size_t>(1, params_.leaf_max_size)) + 1);

    std::vector<int32_t> indices(num_vectors);
    std::iota(indices.begin(), indices.end(), 0);

    tree.root = build_node(tree, indices, 0, num_vectors, rng);
  }
}

int RandomizedKdTree::build_node(SingleTree &tree,
                                 std::vector<int32_t> &indices,
                                 size_t start,
                                 size_t end,
                                 std::mt19937 &rng) {
  const size_t count = end - start;
  if (count <= params_.leaf_max_size) {
    int node_idx = static_cast<int>(tree.nodes.size());
    KdNode leaf;
    leaf.split_dim = -1;
    leaf.indices.assign(indices.begin() + start, indices.begin() + end);
    tree.nodes.push_back(std::move(leaf));
    return node_idx;
  }

  // 1. Calculate variance across all dimensions for this subset
  // Sample up to 100 points if subset is huge to make tree building fast
  const size_t sample_size = std::min<size_t>(count, 100);
  std::vector<std::pair<float, int>> dim_variances(dim_);

  for (size_t d = 0; d < dim_; ++d) {
    float sum = 0.0f;
    float sum_sq = 0.0f;
    for (size_t s = 0; s < sample_size; ++s) {
      int32_t idx = indices[start + s];
      float val = data_[idx * dim_ + d];
      sum += val;
      sum_sq += val * val;
    }
    float mean = sum / static_cast<float>(sample_size);
    float var = (sum_sq / static_cast<float>(sample_size)) - (mean * mean);
    dim_variances[d] = {var, static_cast<int>(d)};
  }

  // Partial sort to find top variance dimensions
  const size_t top_k_dims = std::min(params_.top_variance_dims, dim_);
  std::partial_sort(dim_variances.begin(), dim_variances.begin() + top_k_dims, dim_variances.end(),
                    [](const auto &a, const auto &b) { return a.first > b.first; });

  // 2. Pick split dimension uniformly at random from top candidates
  std::uniform_int_distribution<size_t> dist(0, top_k_dims - 1);
  const int split_dim = dim_variances[dist(rng)].second;

  // 3. Compute median / mean along split_dim to divide into equal partitions
  size_t mid = start + count / 2;
  std::nth_element(indices.begin() + start, indices.begin() + mid, indices.begin() + end,
                   [this, split_dim](int32_t a, int32_t b) {
                     return data_[a * dim_ + split_dim] < data_[b * dim_ + split_dim];
                   });

  const float split_val = data_[indices[mid] * dim_ + split_dim];

  // Allocate current node index
  int node_idx = static_cast<int>(tree.nodes.size());
  tree.nodes.emplace_back(); // placeholder

  int left_child = build_node(tree, indices, start, mid, rng);
  int right_child = build_node(tree, indices, mid, end, rng);

  tree.nodes[node_idx].split_dim = split_dim;
  tree.nodes[node_idx].split_val = split_val;
  tree.nodes[node_idx].left = left_child;
  tree.nodes[node_idx].right = right_child;

  return node_idx;
}

std::vector<SearchResult> RandomizedKdTree::search(
    const std::vector<float> &query,
    size_t top_k,
    const KdSearchParams &search_params) const {
  if (query.size() != dim_) {
    throw std::invalid_argument("Query dimension mismatch in RandomizedKdTree::search");
  }
  return search(query.data(), top_k, search_params);
}

std::vector<SearchResult> RandomizedKdTree::search(
    const float *query,
    size_t top_k,
    const KdSearchParams &search_params) const {
  if (data_ == nullptr || num_vectors_ == 0 || query == nullptr || top_k == 0) {
    return {};
  }

  const size_t safe_top_k = std::min(top_k, num_vectors_);

  // Priority queue for Top-K results: Max-heap of (distance, index)
  std::priority_queue<std::pair<float, int32_t>> result_heap;

  // Priority queue for BBF (Best-Bin-First) search: Min-heap of (distance_to_plane, tree_idx, node_idx)
  struct Branch {
    float dist_sq;
    int tree_idx;
    int node_idx;

    bool operator>(const Branch &other) const noexcept {
      return dist_sq > other.dist_sq;
    }
  };

  std::priority_queue<Branch, std::vector<Branch>, std::greater<Branch>> bbf_queue;

  // Track visited vector IDs to avoid duplicate distance computations
  std::vector<bool> visited(num_vectors_, false);
  size_t checks_done = 0;

  // Seed BBF priority queue with roots of all trees
  for (size_t t = 0; t < trees_.size(); ++t) {
    if (trees_[t].root >= 0) {
      bbf_queue.push({0.0f, static_cast<int>(t), trees_[t].root});
    }
  }

  while (!bbf_queue.empty() && checks_done < search_params.max_checks) {
    Branch cur = bbf_queue.top();
    bbf_queue.pop();

    // If queue distance is already larger than the worst distance in a full top-k heap,
    // in strict exact KD-tree we could prune, but in BBF we continue up to max_checks
    const SingleTree &tree = trees_[cur.tree_idx];
    int cur_node_idx = cur.node_idx;

    // Traverse down the tree greedily until a leaf is hit
    while (cur_node_idx >= 0 && !tree.nodes[cur_node_idx].is_leaf()) {
      const KdNode &node = tree.nodes[cur_node_idx];
      float val = query[node.split_dim];
      float diff = val - node.split_val;
      float diff_sq = diff * diff;

      int closer_child = (diff <= 0.0f) ? node.left : node.right;
      int further_child = (diff <= 0.0f) ? node.right : node.left;

      if (further_child >= 0) {
        bbf_queue.push({diff_sq, cur.tree_idx, further_child});
      }

      cur_node_idx = closer_child;
    }

    // Process leaf node
    if (cur_node_idx >= 0 && tree.nodes[cur_node_idx].is_leaf()) {
      const KdNode &leaf = tree.nodes[cur_node_idx];
      for (int32_t idx : leaf.indices) {
        if (visited[idx]) {
          continue;
        }
        visited[idx] = true;
        ++checks_done;

        const float *vec = data_ + idx * dim_;
        float dist = compute_vector_distance(search_params.metric, vec, query, dim_);

        if (result_heap.size() < safe_top_k) {
          result_heap.emplace(dist, idx);
        } else if (dist < result_heap.top().first) {
          result_heap.pop();
          result_heap.emplace(dist, idx);
        }

        if (checks_done >= search_params.max_checks) {
          break;
        }
      }
    }
  }

  // Extract results sorted ascending by distance
  std::vector<SearchResult> results(result_heap.size());
  for (int i = static_cast<int>(result_heap.size()) - 1; i >= 0; --i) {
    results[i].distance = result_heap.top().first;
    results[i].index = result_heap.top().second;
    result_heap.pop();
  }

  return results;
}

std::vector<std::vector<SearchResult>> RandomizedKdTree::batch_search(
    const float *queries,
    size_t num_queries,
    size_t top_k,
    const KdSearchParams &search_params) const {
  std::vector<std::vector<SearchResult>> all_results(num_queries);
  for (size_t q = 0; q < num_queries; ++q) {
    all_results[q] = search(queries + q * dim_, top_k, search_params);
  }
  return all_results;
}

} // namespace secan
