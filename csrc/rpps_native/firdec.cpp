// A decimating FIR: the filter evaluated only at the output instants (rpps/profile/steps.py KnownTiming with a matched filter
// fused in). out[k] = sum_j h[j] x[idx[k] - j]: one dot product of the taps with the signal a symbol at a time, where filtering
// every sample and keeping every sps-th computes sps times as much. Outputs are independent: they run on several threads.
#include "common.hpp"
#include "parallel.hpp"

#include <algorithm>
#include <cstdint>

namespace rpps {

namespace {

#if defined(__GNUC__) && defined(__x86_64__) && !defined(__clang__)
#define RPPS_CLONES __attribute__((target_clones("avx2,fma", "default")))
#else
#define RPPS_CLONES
#endif

// sum_j h[j] * x[j'] with x walked backwards from `last`, real and imaginary parts apart so the loop vectorises
RPPS_CLONES
static void dot_back(const cf32 *x, const double *h, long long nh, float *re, float *im) {
    double ar[4] = {0, 0, 0, 0}, ai[4] = {0, 0, 0, 0};
    long long j = 0;
    for (; j + 4 <= nh; j += 4)
        for (int i = 0; i < 4; i++) { ar[i] += h[j + i]*x[-(j + i)].real(); ai[i] += h[j + i]*x[-(j + i)].imag(); }
    double sr = (ar[0] + ar[2]) + (ar[1] + ar[3]), si = (ai[0] + ai[2]) + (ai[1] + ai[3]);
    for (; j < nh; j++) { sr += h[j]*x[-j].real(); si += h[j]*x[-j].imag(); }
    *re = static_cast<float>(sr); *im = static_cast<float>(si);
}

py::array_t<cf32> fir_decimate(carray<cf32> x_a, carray<double> h_a, long long start, long long step, long long count, int threads) {
    const long long S = x_a.size(), nh = h_a.size();
    if (nh < 1 || step < 1 || count < 0) throw std::invalid_argument("fir_decimate: bad taps, step or count");
    // Output k is at x[start + k*step]; the filter reaches back nh-1 samples from it
    if (count > 0 && (start - (nh - 1) < 0 || start + (count - 1)*step >= S)) throw std::invalid_argument("fir_decimate: instants outside the signal");
    py::array_t<cf32> out(static_cast<py::ssize_t>(count));
    cf32 *o = out.mutable_data();
    const cf32 *x = x_a.data();
    const double *h = h_a.data();
    constexpr long long CHUNK = 2048;
    {
        py::gil_scoped_release nogil;
        parallel_for((count + CHUNK - 1)/CHUNK, threads, [&](long long blk) {
            const long long end = std::min(count, (blk + 1)*CHUNK);
            for (long long k = blk*CHUNK; k < end; k++) {
                float re, im;
                dot_back(x + start + k*step, h, nh, &re, &im);
                o[k] = cf32(re, im);
            }
        });
    }
    return out;
}

} // namespace

void register_firdec(py::module_ &m) {
    m.def("fir_decimate", &fir_decimate,
          "fir_decimate(x complex64, taps float64, start, step, count, threads) -> out[k] = sum_j taps[j] x[start + k*step - j] (complex64)",
          py::arg("x"), py::arg("taps"), py::arg("start"), py::arg("step"), py::arg("count"), py::arg("threads") = 0);
}

} // namespace rpps
