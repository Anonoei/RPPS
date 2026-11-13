// Direct-sequence despreading: correlate blocks of chips with a set of spreading sequences
//
// Mirrors rpps/spread/despread.py: the Python stage does the block bookkeeping, the kernel the inner product.
#include "common.hpp"

#include <vector>

namespace rpps {

namespace {

// x: n*N*sps samples, chips: (M, N) of +/-1. out[p, m] = sum_k chips[m, k] * sum_s x[(p*N + k)*sps + s] / (N*sps).
// Sums in double (numpy upcasts the matrix product the same way); the output keeps the sample precision.
template <typename T>
py::array_t<std::complex<T>> despread(carray<std::complex<T>> x, carray<double> chips, int sps) {
    if (sps < 1)
        throw std::invalid_argument("despread: sps must be >= 1");
    if (chips.ndim() != 2)
        throw std::invalid_argument("despread: chips must be (count, length)");
    const std::size_t M = static_cast<std::size_t>(chips.shape(0)), N = static_cast<std::size_t>(chips.shape(1));
    const std::size_t S = static_cast<std::size_t>(sps);
    const std::size_t total = static_cast<std::size_t>(x.size());
    if (N == 0 || total % (N*S) != 0)
        throw std::invalid_argument("despread: x must hold whole code periods");
    const std::size_t n = total / (N*S);
    auto out = py::array_t<std::complex<T>>({static_cast<py::ssize_t>(n), static_cast<py::ssize_t>(M)});
    const std::complex<T> *in = x.data();
    const double *c = chips.data();
    std::complex<T> *o = out.mutable_data();
    {
        py::gil_scoped_release nogil;
        std::vector<double> chip_sum_re(N), chip_sum_im(N);
        const double norm = 1.0 / static_cast<double>(N*S);
        for (std::size_t p = 0; p < n; p++) {
            // Samples of each chip summed once, then each sequence takes its inner product
            for (std::size_t k = 0; k < N; k++) {
                double re = 0.0, im = 0.0;
                const std::complex<T> *b = in + (p*N + k)*S;
                for (std::size_t s = 0; s < S; s++) {
                    re += static_cast<double>(b[s].real());
                    im += static_cast<double>(b[s].imag());
                }
                chip_sum_re[k] = re;
                chip_sum_im[k] = im;
            }
            for (std::size_t m = 0; m < M; m++) {
                double re = 0.0, im = 0.0;
                const double *cm = c + m*N;
                for (std::size_t k = 0; k < N; k++) {
                    re += cm[k]*chip_sum_re[k];
                    im += cm[k]*chip_sum_im[k];
                }
                o[p*M + m] = std::complex<T>(static_cast<T>(re*norm), static_cast<T>(im*norm));
            }
        }
    }
    return out;
}

} // namespace

void register_spread(py::module_ &m) {
    const char *doc = "despread(x, chips, sps) -> (periods, count) correlations of each code period with each sequence";
    m.def("despread", &despread<float>, doc, py::arg("x").noconvert(), py::arg("chips"), py::arg("sps"));
    m.def("despread", &despread<double>, doc, py::arg("x"), py::arg("chips"), py::arg("sps"));
}

} // namespace rpps
