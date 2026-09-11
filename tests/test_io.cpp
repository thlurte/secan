#include "secan/utils/io.h"
#include "test_utils.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

namespace fs = std::filesystem;

void test_fvecs_roundtrip() {
  const std::string test_path = "/tmp/test_secan_roundtrip.fvecs";
  
  secan::FloatDataset original;
  original.num_vectors = 4;
  original.dim = 3;
  original.data = {
      1.0f, 2.0f, 3.0f,
      4.5f, -5.5f, 6.25f,
      0.0f, 100.0f, -100.0f,
      3.14159f, 2.71828f, 1.41421f
  };

  secan::write_fvecs(test_path, original);
  CHECK(fs::exists(test_path));

  // Test standard stream loader
  secan::FloatDataset loaded = secan::load_fvecs(test_path);
  CHECK(loaded.num_vectors == original.num_vectors);
  CHECK(loaded.dim == original.dim);
  for (size_t i = 0; i < original.data.size(); ++i) {
    CHECK(loaded.data[i] == original.data[i]);
  }

  // Test memory-mapped loader
  {
    secan::MmapFloatDataset mmap_ds = secan::load_fvecs_mmap(test_path);
    CHECK(mmap_ds.is_open());
    CHECK(mmap_ds.size() == original.num_vectors);
    CHECK(mmap_ds.dim() == original.dim);
    for (size_t i = 0; i < original.num_vectors; ++i) {
      const float *orig_vec = original.get(i);
      const float *mmap_vec = mmap_ds.get(i);
      for (size_t d = 0; d < original.dim; ++d) {
        CHECK(mmap_vec[d] == orig_vec[d]);
      }
    }
  }

  fs::remove(test_path);
}

void test_bvecs_roundtrip() {
  const std::string test_path = "/tmp/test_secan_roundtrip.bvecs";

  secan::ByteDataset original;
  original.num_vectors = 3;
  original.dim = 4;
  original.data = {
      0, 128, 255, 42,
      1, 2, 3, 4,
      250, 251, 252, 253
  };

  secan::write_bvecs(test_path, original);
  CHECK(fs::exists(test_path));

  secan::ByteDataset loaded = secan::load_bvecs(test_path);
  CHECK(loaded.num_vectors == original.num_vectors);
  CHECK(loaded.dim == original.dim);
  for (size_t i = 0; i < original.data.size(); ++i) {
    CHECK(loaded.data[i] == original.data[i]);
  }

  fs::remove(test_path);
}

void test_ivecs_roundtrip() {
  const std::string test_path = "/tmp/test_secan_roundtrip.ivecs";

  secan::IntDataset original;
  original.num_vectors = 3;
  original.dim = 5;
  original.data = {
      10, 20, 30, 40, 50,
      -1, -2, -3, -4, -5,
      100000, 200000, 300000, 400000, 500000
  };

  secan::write_ivecs(test_path, original);
  CHECK(fs::exists(test_path));

  secan::IntDataset loaded = secan::load_ivecs(test_path);
  CHECK(loaded.num_vectors == original.num_vectors);
  CHECK(loaded.dim == original.dim);
  for (size_t i = 0; i < original.data.size(); ++i) {
    CHECK(loaded.data[i] == original.data[i]);
  }

  fs::remove(test_path);
}

void test_io_error_handling() {
  // Non-existent file
  bool caught_missing = false;
  try {
    secan::load_fvecs("/tmp/non_existent_file_xyz_123.fvecs");
  } catch (const std::runtime_error &) {
    caught_missing = true;
  }
  CHECK(caught_missing);

  // Corrupt / truncated file
  const std::string corrupt_path = "/tmp/test_corrupt.fvecs";
  std::ofstream out(corrupt_path, std::ios::binary);
  int32_t bad_dim = 128;
  out.write(reinterpret_cast<char *>(&bad_dim), sizeof(int32_t));
  float half_vec[10] = {0.0f}; // Should be 128 floats, only wrote 10
  out.write(reinterpret_cast<char *>(half_vec), sizeof(half_vec));
  out.close();

  bool caught_corrupt = false;
  try {
    secan::load_fvecs(corrupt_path);
  } catch (const std::runtime_error &) {
    caught_corrupt = true;
  }
  CHECK(caught_corrupt);

  fs::remove(corrupt_path);
}

void test_sift1m_if_available() {
  const std::string sift_path = "data/sift1m/sift_base.fvecs";
  if (fs::exists(sift_path)) {
    std::cout << "Found SIFT1M base dataset at " << sift_path << ", validating header..." << std::endl;
    auto ds = secan::load_fvecs(sift_path);
    CHECK(ds.num_vectors == 1000000);
    CHECK(ds.dim == 128);

    auto mmap_ds = secan::load_fvecs_mmap(sift_path);
    CHECK(mmap_ds.size() == 1000000);
    CHECK(mmap_ds.dim() == 128);
    std::cout << "SIFT1M base header and mmap validated successfully!" << std::endl;
  } else {
    std::cout << "[NOTE] SIFT1M base dataset not yet staged in data/sift1m/. Skipping large dataset check." << std::endl;
  }
}

int main() {
  test_fvecs_roundtrip();
  test_bvecs_roundtrip();
  test_ivecs_roundtrip();
  test_io_error_handling();
  // Test 10k dataset loading throughput
  {
    const std::string path = "data/test_10k.fvecs";
    if (fs::exists(path)) {
      auto ds = secan::load_fvecs(path);
      CHECK(ds.num_vectors == 10000);
      CHECK(ds.dim == 128);

      auto mmap_ds = secan::load_fvecs_mmap(path);
      CHECK(mmap_ds.size() == 10000);
      CHECK(mmap_ds.dim() == 128);
    }
  }
  test_sift1m_if_available();
  return report_results("io_tests");
}
