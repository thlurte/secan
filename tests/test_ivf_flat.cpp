#include "secan/index/ivf_flat.h"
#include "test_utils.h"
#include <cstdint>
#include <vector>

// Test 1: Verify Cache-Line Alignment to prevent False Sharing (Pikus Ch 5)
void test_inverted_list_alignment() {
  std::cout << "\n--- Testing InvertedList Cache Alignment ---" << std::endl;

  // alignas(64) ensures each InvertedList instance sits on its own 64-byte
  // cache line
  CHECK(alignof(secan::InvertedList) >= 64);
  CHECK(sizeof(secan::InvertedList) >= 64);
}

// Test 2: Verify InvertedList Operations & Contiguous Memory Layout
void test_inverted_list_operations() {
  std::cout << "\n--- Testing InvertedList Operations ---" << std::endl;

  secan::InvertedList list;
  CHECK(list.empty());
  CHECK(list.size() == 0);

  size_t dim = 4;
  list.reserve(3, dim);

  float v0[] = {1.0f, 2.0f, 3.0f, 4.0f};
  float v1[] = {5.0f, 6.0f, 7.0f, 8.0f};
  float v2[] = {9.0f, 10.0f, 11.0f, 12.0f};

  list.add(100, v0, dim);
  list.add(101, v1, dim);
  list.add(102, v2, dim);

  CHECK(!list.empty());
  CHECK(list.size() == 3);
  CHECK(list.ids.size() == 3);
  CHECK(list.data.size() == 12); // 3 vectors * 4 floats

  // Verify IDs
  CHECK(list.ids[0] == 100);
  CHECK(list.ids[1] == 101);
  CHECK(list.ids[2] == 102);

  // Verify Contiguous Memory Layout (Sequential SIMD Streaming)
  // Check that vectors are packed sequentially without gaps:
  CHECK(list.data[0] == 1.0f && list.data[3] == 4.0f);
  CHECK(list.data[4] == 5.0f && list.data[7] == 8.0f);
  CHECK(list.data[8] == 9.0f && list.data[11] == 12.0f);

  // Test clear
  list.clear();
  CHECK(list.empty());
  CHECK(list.size() == 0);
  CHECK(list.data.empty());
}

// Test 3: Verify IvfFlatIndex Initialization & Metadata
void test_ivf_flat_index_init() {
  std::cout << "\n--- Testing IvfFlatIndex Initialization ---" << std::endl;

  size_t dim = 128;
  size_t nlist = 16;
  secan::IvfFlatIndex index(dim, nlist, secan::MetricType::L2);

  CHECK(index.get_dim() == 128);
  CHECK(index.get_nlist() == 16);
  CHECK(!index.is_trained());
  CHECK(index.total_vectors() == 0);
  CHECK(index.get_lists().size() == 16);

  // Centroid buffer must be pre-allocated to nlist * dim floats
  CHECK(index.get_centroids() != nullptr);
}

// Test 4: End-to-End Train, Add & Search Verification
void test_ivf_flat_train_add_search() {
  std::cout << "\n--- Testing IvfFlatIndex Train, Add & Multi-Probe Search ---" << std::endl;

  const size_t dim = 4;
  const size_t nlist = 2;
  const size_t n_vecs = 6;

  secan::IvfFlatIndex index(dim, nlist, secan::MetricType::L2);

  std::vector<float> data = {
      0.1f, 0.0f, 0.1f, 0.0f,    // ID 0
      0.2f, 0.1f, 0.0f, 0.1f,    // ID 1
      0.0f, 0.1f, 0.2f, 0.0f,    // ID 2
      10.0f, 10.1f, 10.0f, 9.9f, // ID 3
      10.2f, 10.0f, 10.1f, 10.0f,// ID 4
      9.9f, 10.0f, 10.2f, 10.1f  // ID 5
  };
  std::vector<int32_t> ids = {0, 1, 2, 3, 4, 5};

  index.train(n_vecs, data.data(), 10);
  CHECK(index.is_trained());

  index.add(n_vecs, ids.data(), data.data());
  CHECK(index.total_vectors() == n_vecs);

  // Search for query close to cluster 1
  float query[] = {0.1f, 0.05f, 0.1f, 0.0f};
  auto results = index.search(query, 3, 2);

  CHECK(results.size() == 3);
  CHECK(results[0].id == 0);
  CHECK(results[0].distance <= results[1].distance);

  // Batch search verification
  std::vector<float> queries = {
      0.1f, 0.05f, 0.1f, 0.0f,
      10.0f, 10.0f, 10.0f, 10.0f
  };
  auto batch_res = index.batch_search(2, queries.data(), 2, 2);
  CHECK(batch_res.size() == 2);
  CHECK(batch_res[0].size() == 2);
  CHECK(batch_res[0][0].id == 0);
  CHECK(batch_res[1][0].id >= 3);
}

// Test 5: Inverted List Statistics & Skew Diagnostics
void test_ivf_flat_stats() {
  std::cout << "\n--- Testing IvfFlatIndex List Distribution Diagnostics ---" << std::endl;

  const size_t dim = 4;
  const size_t nlist = 4;
  secan::IvfFlatIndex index(dim, nlist, secan::MetricType::L2);

  std::vector<float> data = {
      0.0f, 0.0f, 0.0f, 0.0f,
      0.1f, 0.1f, 0.1f, 0.1f,
      5.0f, 5.0f, 5.0f, 5.0f,
      5.1f, 5.1f, 5.1f, 5.1f
  };
  std::vector<int32_t> ids = {0, 1, 2, 3};

  index.train(4, data.data(), 5);
  index.add(4, ids.data(), data.data());

  auto stats = index.get_list_stats();
  CHECK(stats.total_vectors == 4);
  CHECK(stats.mean_list_size == 1.0);
  CHECK(stats.max_list_size >= 1);
}

int main() {
  print_clock_overhead();

  test_inverted_list_alignment();
  test_inverted_list_operations();
  test_ivf_flat_index_init();
  test_ivf_flat_train_add_search();
  test_ivf_flat_stats();

  std::cout << "\n========================================\n";
  std::cout << "Tests run: " << g_tests_run << "\n";
  std::cout << "Tests failed: " << g_tests_failed << "\n";
  std::cout << "========================================\n";

  return g_tests_failed == 0 ? 0 : 1;
}
