// Drill 1: The Pointer Offset Reflex (Zero-Indexing Flattener)
// Completed: 2026-09-15
// Concept: 2D row-major matrix flattened into 1D memory array.

#include <cstddef>
#include <cassert>
#include <iostream>

// Return pointer to start of vector i (address math)
const float* get_vector(const float* flat_data, size_t i, size_t dim) {
    return flat_data + (i * dim);
}

// Return j-th element of vector i
float get_element(const float* flat_data, size_t i, size_t j, size_t dim) {
    return flat_data[i * dim + j];
}

// Alternative idiomatic formulation reusing get_vector:
float get_element_clean(const float* flat_data, size_t i, size_t j, size_t dim) {
    const float* row = get_vector(flat_data, i, dim);
    return row[j];
}

int main() {
    // 3 vectors of dimension 4
    // Vector 0: [ 1,  2,  3,  4]
    // Vector 1: [ 5,  6,  7,  8]
    // Vector 2: [ 9, 10, 11, 12]
    float data[12] = {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f,
        9.0f, 10.0f, 11.0f, 12.0f
    };

    const float* v1 = get_vector(data, 1, 4);
    assert(v1 == data + 4);
    assert(*v1 == 5.0f);

    float elem = get_element(data, 2, 1, 4);
    assert(elem == 10.0f);

    float elem_clean = get_element_clean(data, 2, 1, 4);
    assert(elem_clean == 10.0f);

    std::cout << "Drill 1 tests passed successfully!" << std::endl;
    return 0;
}
