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

int main() {
  print_clock_overhead();

  test_inverted_list_alignment();
  test_inverted_list_operations();
  test_ivf_flat_index_init();

  std::cout << "\n========================================\n";
  std::cout << "Tests run: " << g_tests_run << "\n";
  std::cout << "Tests failed: " << g_tests_failed << "\n";
  std::cout << "========================================\n";

  return g_tests_failed == 0 ? 0 : 1;
}
