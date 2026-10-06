#include "secan/index/flat.h"
#include "secan/index/ivf_flat.h"
#include "secan/search/distance_avx2.h"
#include "test_utils.h"
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

void test_inverted_list_alignment() {
  std::cout << "\n--- Testing InvertedList Cache Alignment ---" << std::endl;
  CHECK(alignof(secan::InvertedList) >= 64);
  CHECK(sizeof(secan::InvertedList) >= 64);
}

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
  CHECK(list.data.size() == 12);

  CHECK(list.ids[0] == 100);
  CHECK(list.ids[1] == 101);
  CHECK(list.ids[2] == 102);

  CHECK(list.data[0] == 1.0f && list.data[3] == 4.0f);
  CHECK(list.data[4] == 5.0f && list.data[7] == 8.0f);
  CHECK(list.data[8] == 9.0f && list.data[11] == 12.0f);

  list.clear();
  CHECK(list.empty());
  CHECK(list.size() == 0);
  CHECK(list.data.empty());
}

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
  CHECK(index.get_centroids() != nullptr);
}

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

  float query[] = {0.1f, 0.05f, 0.1f, 0.0f};
  auto results = index.search(query, 3, 2);

  CHECK(results.size() == 3);
  CHECK(results[0].id == 0);
  CHECK(results[0].distance <= results[1].distance);

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

void test_ivf_flat_ip_separable() {
  std::cout << "\n--- Testing IVF MetricType::IP Separability ---" << std::endl;

  const size_t dim = 8;
  const size_t nlist = 2;
  const size_t n_vecs = 6;

  // Two separable groups in different directions
  std::vector<float> data = {
      1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
      0.9f, 0.1f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
      0.8f, 0.2f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,

      0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f,
      0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.1f, 0.9f,
      0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.2f, 0.8f
  };
  std::vector<int32_t> ids = {0, 1, 2, 3, 4, 5};

  secan::IvfFlatIndex index(dim, nlist, secan::MetricType::IP);
  index.train(n_vecs, data.data(), 15);
  index.add(n_vecs, ids.data(), data.data());

  // Check that both inverted lists received items (not all in list 0)
  CHECK(!index.get_lists()[0].empty());
  CHECK(!index.get_lists()[1].empty());
  CHECK(index.get_lists()[0].size() == 3);
  CHECK(index.get_lists()[1].size() == 3);
}

void test_ivf_flat_oracle_parity_l2_and_ip() {
  std::cout << "\n--- Testing IVF nprobe==nlist vs Flat2D Exact Scan Parity ---" << std::endl;

  const size_t dim = 32;
  const size_t n_vecs = 200;
  const size_t nlist = 8;
  const size_t k = 5;

  std::mt19937 rng(1337);
  std::normal_distribution<float> dist(0.0f, 1.0f);

  secan::FloatDataset dataset;
  dataset.dim = dim;
  dataset.num_vectors = n_vecs;
  dataset.data.resize(n_vecs * dim);
  for (float &val : dataset.data) val = dist(rng);

  std::vector<int32_t> ids(n_vecs);
  for (size_t i = 0; i < n_vecs; ++i) ids[i] = static_cast<int32_t>(i);

  std::vector<float> query(dim);
  for (float &val : query) val = dist(rng);

  // Test L2 parity
  {
    secan::Flat2DIndex flat_l2(dim, secan::MetricType::L2);
    flat_l2.add(dataset);
    auto flat_res = flat_l2.search(query.data(), k);

    secan::IvfFlatIndex ivf_l2(dim, nlist, secan::MetricType::L2);
    ivf_l2.train(n_vecs, dataset.data.data(), 10);
    ivf_l2.add(n_vecs, ids.data(), dataset.data.data());
    auto ivf_res = ivf_l2.search(query.data(), k, nlist);

    CHECK(ivf_res.size() == k);
    for (size_t i = 0; i < k; ++i) {
      CHECK(ivf_res[i].id == flat_res[i].index);
      CHECK_NEAR(ivf_res[i].distance, flat_res[i].distance, 1e-3f);
    }
  }

  // Test IP parity
  {
    secan::Flat2DIndex flat_ip(dim, secan::MetricType::IP);
    flat_ip.add(dataset);
    auto flat_res = flat_ip.search(query.data(), k);

    secan::IvfFlatIndex ivf_ip(dim, nlist, secan::MetricType::IP);
    ivf_ip.train(n_vecs, dataset.data.data(), 10);
    ivf_ip.add(n_vecs, ids.data(), dataset.data.data());
    auto ivf_res = ivf_ip.search(query.data(), k, nlist);

    CHECK(ivf_res.size() == k);
    for (size_t i = 0; i < k; ++i) {
      CHECK(ivf_res[i].id == flat_res[i].index);
      CHECK_NEAR(ivf_res[i].distance, flat_res[i].distance, 1e-3f);
    }
  }
}

void test_ivf_flat_spherical_kmeans_exact_norm() {
  std::cout << "\n--- Testing Spherical K-Means Exact Normalization ---" << std::endl;

  const size_t dim = 16;
  const size_t nlist = 4;
  const size_t n_vecs = 40;

  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(2.0f, 10.0f); // Non-unit vectors

  std::vector<float> data(n_vecs * dim);
  for (float &val : data) val = dist(rng);

  secan::IvfFlatIndex index(dim, nlist, secan::MetricType::Cosine);
  index.train(n_vecs, data.data(), 10);

  const float *centroids = index.get_centroids();
  for (size_t k = 0; k < nlist; ++k) {
    const float *c = centroids + k * dim;
    float norm_sq = 0.0f;
    for (size_t d = 0; d < dim; ++d) {
      norm_sq += c[d] * c[d];
    }
    float norm = std::sqrt(norm_sq);
    // Strict exact norm condition: | ||c||_2 - 1 | < 1e-5
    CHECK(std::abs(norm - 1.0f) < 1e-5f);
  }
}

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
  test_inverted_list_alignment();
  test_inverted_list_operations();
  test_ivf_flat_index_init();
  test_ivf_flat_train_add_search();
  test_ivf_flat_ip_separable();
  test_ivf_flat_oracle_parity_l2_and_ip();
  test_ivf_flat_spherical_kmeans_exact_norm();
  test_ivf_flat_stats();
  return report_results("ivf_flat_tests");
}
