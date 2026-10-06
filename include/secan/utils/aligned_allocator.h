#pragma once

#include <cstddef>
#include <cstdlib>
#include <limits>
#include <new>
#include <vector>

namespace secan {

template <typename T, size_t Alignment = 64>
class AlignedAllocator {
public:
    using value_type = T;
    using size_type = size_t;
    using difference_type = ptrdiff_t;

    AlignedAllocator() noexcept = default;
    template <typename U>
    AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}

    T* allocate(size_t n) {
        if (n == 0) return nullptr;
        if (n > std::numeric_limits<size_t>::max() / sizeof(T)) {
            throw std::bad_array_new_length();
        }
        size_t bytes = n * sizeof(T);
        size_t alloc_bytes = (bytes + Alignment - 1) & ~(Alignment - 1);
#if defined(_MSC_VER)
        void* ptr = _aligned_malloc(alloc_bytes, Alignment);
        if (!ptr) throw std::bad_alloc();
#else
        void* ptr = nullptr;
        int res = posix_memalign(&ptr, Alignment, alloc_bytes);
        if (res != 0 || !ptr) throw std::bad_alloc();
#endif
        return static_cast<T*>(ptr);
    }

    void deallocate(T* p, size_t) noexcept {
        if (!p) return;
#if defined(_MSC_VER)
        _aligned_free(p);
#else
        free(p);
#endif
    }

    template <typename U>
    struct rebind {
        using other = AlignedAllocator<U, Alignment>;
    };

    bool operator==(const AlignedAllocator&) const noexcept { return true; }
    bool operator!=(const AlignedAllocator&) const noexcept { return false; }
};

using AlignedFloatVector = std::vector<float, AlignedAllocator<float, 64>>;

} // namespace secan
