#include "secan/utils/io.h"

#include <fcntl.h>
#include <fstream>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace secan {

FloatDataset load_fvecs(const std::string &path) {
  std::ifstream file(path, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open fvecs file: " + path);
  }

  int32_t d = 0;
  file.read(reinterpret_cast<char *>(&d), sizeof(int32_t));
  if (file.gcount() != sizeof(int32_t) || d <= 0) {
    throw std::runtime_error("Invalid or empty fvecs header in: " + path);
  }

  file.seekg(0, std::ios::end);
  const size_t file_size = static_cast<size_t>(file.tellg());
  const size_t record_size = sizeof(int32_t) + static_cast<size_t>(d) * sizeof(float);

  if (file_size % record_size != 0) {
    throw std::runtime_error("Corrupted fvecs file size in: " + path);
  }

  const size_t num_vectors = file_size / record_size;
  FloatDataset ds;
  ds.num_vectors = num_vectors;
  ds.dim = static_cast<size_t>(d);
  ds.data.resize(num_vectors * ds.dim);

  file.seekg(0, std::ios::beg);
  for (size_t i = 0; i < num_vectors; ++i) {
    int32_t cur_d = 0;
    file.read(reinterpret_cast<char *>(&cur_d), sizeof(int32_t));
    if (cur_d != d) {
      throw std::runtime_error("Mismatched vector dimension at index " +
                               std::to_string(i) + " in: " + path);
    }
    file.read(reinterpret_cast<char *>(ds.data.data() + i * ds.dim),
              static_cast<std::streamsize>(ds.dim * sizeof(float)));
  }

  return ds;
}

ByteDataset load_bvecs(const std::string &path) {
  std::ifstream file(path, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open bvecs file: " + path);
  }

  int32_t d = 0;
  file.read(reinterpret_cast<char *>(&d), sizeof(int32_t));
  if (file.gcount() != sizeof(int32_t) || d <= 0) {
    throw std::runtime_error("Invalid or empty bvecs header in: " + path);
  }

  file.seekg(0, std::ios::end);
  const size_t file_size = static_cast<size_t>(file.tellg());
  const size_t record_size = sizeof(int32_t) + static_cast<size_t>(d) * sizeof(uint8_t);

  if (file_size % record_size != 0) {
    throw std::runtime_error("Corrupted bvecs file size in: " + path);
  }

  const size_t num_vectors = file_size / record_size;
  ByteDataset ds;
  ds.num_vectors = num_vectors;
  ds.dim = static_cast<size_t>(d);
  ds.data.resize(num_vectors * ds.dim);

  file.seekg(0, std::ios::beg);
  for (size_t i = 0; i < num_vectors; ++i) {
    int32_t cur_d = 0;
    file.read(reinterpret_cast<char *>(&cur_d), sizeof(int32_t));
    if (cur_d != d) {
      throw std::runtime_error("Mismatched vector dimension at index " +
                               std::to_string(i) + " in: " + path);
    }
    file.read(reinterpret_cast<char *>(ds.data.data() + i * ds.dim),
              static_cast<std::streamsize>(ds.dim * sizeof(uint8_t)));
  }

  return ds;
}

IntDataset load_ivecs(const std::string &path) {
  std::ifstream file(path, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open ivecs file: " + path);
  }

  int32_t d = 0;
  file.read(reinterpret_cast<char *>(&d), sizeof(int32_t));
  if (file.gcount() != sizeof(int32_t) || d <= 0) {
    throw std::runtime_error("Invalid or empty ivecs header in: " + path);
  }

  file.seekg(0, std::ios::end);
  const size_t file_size = static_cast<size_t>(file.tellg());
  const size_t record_size = sizeof(int32_t) + static_cast<size_t>(d) * sizeof(int32_t);

  if (file_size % record_size != 0) {
    throw std::runtime_error("Corrupted ivecs file size in: " + path);
  }

  const size_t num_vectors = file_size / record_size;
  IntDataset ds;
  ds.num_vectors = num_vectors;
  ds.dim = static_cast<size_t>(d);
  ds.data.resize(num_vectors * ds.dim);

  file.seekg(0, std::ios::beg);
  for (size_t i = 0; i < num_vectors; ++i) {
    int32_t cur_d = 0;
    file.read(reinterpret_cast<char *>(&cur_d), sizeof(int32_t));
    if (cur_d != d) {
      throw std::runtime_error("Mismatched vector dimension at index " +
                               std::to_string(i) + " in: " + path);
    }
    file.read(reinterpret_cast<char *>(ds.data.data() + i * ds.dim),
              static_cast<std::streamsize>(ds.dim * sizeof(int32_t)));
  }

  return ds;
}

void write_fvecs(const std::string &path, const FloatDataset &ds) {
  std::ofstream file(path, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open file for writing: " + path);
  }

  const int32_t d = static_cast<int32_t>(ds.dim);
  for (size_t i = 0; i < ds.num_vectors; ++i) {
    file.write(reinterpret_cast<const char *>(&d), sizeof(int32_t));
    file.write(reinterpret_cast<const char *>(ds.get(i)),
               static_cast<std::streamsize>(ds.dim * sizeof(float)));
  }
}

void write_bvecs(const std::string &path, const ByteDataset &ds) {
  std::ofstream file(path, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open file for writing: " + path);
  }

  const int32_t d = static_cast<int32_t>(ds.dim);
  for (size_t i = 0; i < ds.num_vectors; ++i) {
    file.write(reinterpret_cast<const char *>(&d), sizeof(int32_t));
    file.write(reinterpret_cast<const char *>(ds.get(i)),
               static_cast<std::streamsize>(ds.dim * sizeof(uint8_t)));
  }
}

void write_ivecs(const std::string &path, const IntDataset &ds) {
  std::ofstream file(path, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open file for writing: " + path);
  }

  const int32_t d = static_cast<int32_t>(ds.dim);
  for (size_t i = 0; i < ds.num_vectors; ++i) {
    file.write(reinterpret_cast<const char *>(&d), sizeof(int32_t));
    file.write(reinterpret_cast<const char *>(ds.get(i)),
               static_cast<std::streamsize>(ds.dim * sizeof(int32_t)));
  }
}

MmapFloatDataset::~MmapFloatDataset() {
  close();
}

MmapFloatDataset::MmapFloatDataset(MmapFloatDataset &&other) noexcept
    : mapped_data_(other.mapped_data_),
      file_size_(other.file_size_),
      num_vectors_(other.num_vectors_),
      dim_(other.dim_),
      record_size_(other.record_size_),
      fd_(other.fd_) {
  other.mapped_data_ = nullptr;
  other.file_size_ = 0;
  other.num_vectors_ = 0;
  other.dim_ = 0;
  other.record_size_ = 0;
  other.fd_ = -1;
}

MmapFloatDataset &MmapFloatDataset::operator=(MmapFloatDataset &&other) noexcept {
  if (this != &other) {
    close();
    mapped_data_ = other.mapped_data_;
    file_size_ = other.file_size_;
    num_vectors_ = other.num_vectors_;
    dim_ = other.dim_;
    record_size_ = other.record_size_;
    fd_ = other.fd_;

    other.mapped_data_ = nullptr;
    other.file_size_ = 0;
    other.num_vectors_ = 0;
    other.dim_ = 0;
    other.record_size_ = 0;
    other.fd_ = -1;
  }
  return *this;
}

const float *MmapFloatDataset::get(size_t index) const noexcept {
  return reinterpret_cast<const float *>(
      static_cast<const char *>(mapped_data_) + index * record_size_ + sizeof(int32_t));
}

void MmapFloatDataset::close() {
  if (mapped_data_ != nullptr && mapped_data_ != MAP_FAILED) {
    ::munmap(mapped_data_, file_size_);
    mapped_data_ = nullptr;
  }
  if (fd_ >= 0) {
    ::close(fd_);
    fd_ = -1;
  }
  file_size_ = 0;
  num_vectors_ = 0;
  dim_ = 0;
  record_size_ = 0;
}

MmapFloatDataset MmapFloatDataset::open(const std::string &path) {
  int fd = ::open(path.c_str(), O_RDONLY);
  if (fd < 0) {
    throw std::runtime_error("Failed to open file for mmap: " + path);
  }

  struct stat sb {};
  if (::fstat(fd, &sb) < 0) {
    ::close(fd);
    throw std::runtime_error("Failed to fstat file: " + path);
  }

  const size_t file_size = static_cast<size_t>(sb.st_size);
  if (file_size < sizeof(int32_t)) {
    ::close(fd);
    throw std::runtime_error("File too small to be valid fvecs: " + path);
  }

  int32_t d = 0;
  if (::pread(fd, &d, sizeof(int32_t), 0) != sizeof(int32_t) || d <= 0) {
    ::close(fd);
    throw std::runtime_error("Invalid fvecs header dimension: " + path);
  }

  const size_t record_size = sizeof(int32_t) + static_cast<size_t>(d) * sizeof(float);
  if (file_size % record_size != 0) {
    ::close(fd);
    throw std::runtime_error("File size is not an exact multiple of record size: " + path);
  }

  void *addr = ::mmap(nullptr, file_size, PROT_READ, MAP_SHARED, fd, 0);
  if (addr == MAP_FAILED) {
    ::close(fd);
    throw std::runtime_error("Failed to mmap file: " + path);
  }

  // Advise kernel of sequential access pattern if desired
  #ifdef POSIX_MADV_SEQUENTIAL
  ::posix_madvise(addr, file_size, POSIX_MADV_SEQUENTIAL);
  #endif

  MmapFloatDataset ds;
  ds.mapped_data_ = addr;
  ds.file_size_ = file_size;
  ds.dim_ = static_cast<size_t>(d);
  ds.record_size_ = record_size;
  ds.num_vectors_ = file_size / record_size;
  ds.fd_ = fd;

  return ds;
}

MmapFloatDataset load_fvecs_mmap(const std::string &path) {
  return MmapFloatDataset::open(path);
}

} // namespace secan
