// Shared helpers for the rpps._native kernels
//
// Kernels are flat functions: arrays and loop state in, output and new loop state out. The Python stage keeps its
// state object and the block-size independence; the kernel only runs the per-sample loop, with the GIL released.
#pragma once

#include <pybind11/pybind11.h> // Python.h first: it sets the feature macros
#include <pybind11/numpy.h>

#include <complex>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <string>
#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace py = pybind11;

namespace rpps {

using cf32 = std::complex<float>;
using cf64 = std::complex<double>;

// Index of the lowest set bit of x (x != 0)
inline int ctz64(uint64_t x) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_ctzll(x);
#elif defined(_MSC_VER) && defined(_M_X64)
    unsigned long i;
    _BitScanForward64(&i, x);
    return static_cast<int>(i);
#else
    int i = 0;
    while (!(x & 1)) { x >>= 1; i++; }
    return i;
#endif
}

template <typename T>
using carray = py::array_t<T, py::array::c_style | py::array::forcecast>;

// An uninitialised 1-D numpy array of n items
template <typename T>
inline py::array_t<T> empty(std::size_t n) {
    return py::array_t<T>(static_cast<py::ssize_t>(n));
}

inline void check_shape(const py::array &a, std::initializer_list<py::ssize_t> shape, const char *name) {
    if (a.ndim() != static_cast<py::ssize_t>(shape.size()))
        throw std::invalid_argument(std::string(name) + ": wrong number of dimensions");
    py::ssize_t i = 0;
    for (auto s : shape) {
        if (s >= 0 && a.shape(i) != s)
            throw std::invalid_argument(std::string(name) + ": wrong shape");
        i++;
    }
}

void register_sync(py::module_ &m);
void register_filters(py::module_ &m);
void register_trellis(py::module_ &m);
void register_fec(py::module_ &m);
void register_spread(py::module_ &m);
void register_demap(py::module_ &m);
void register_framesync(py::module_ &m);
void register_firdec(py::module_ &m);
void register_carrier(py::module_ &m);
void register_resample(py::module_ &m);
void register_analysis(py::module_ &m);
void register_polar(py::module_ &m);
void register_bits(py::module_ &m);

} // namespace rpps
