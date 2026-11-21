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
#define RPPS_CLONES __attribute__((target_clones("avx2", "default")))
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
    // About 128k multiply-adds a chunk: a long filter spreads a short block over more threads
    const long long CHUNK = std::max(64LL, std::min(2048LL, (1LL << 17)/nh));
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

// Polyphase interpolation: out[k*R + p] = sum_j phases[p][j] x[start + k - j], every phase of an input sample in one pass
// (what zero stuffing by R and filtering gives, without the zeros)
py::array_t<cf32> fir_interpolate(carray<cf32> x_a, carray<double> ph_a, long long start, long long count, int threads) {
    check_shape(ph_a, {-1, -1}, "phases");
    const long long S = x_a.size(), R = ph_a.shape(0), L = ph_a.shape(1);
    if (R < 1 || L < 1 || count < 0) throw std::invalid_argument("fir_interpolate: bad phases or count");
    if (count > 0 && (start - (L - 1) < 0 || start + count - 1 >= S)) throw std::invalid_argument("fir_interpolate: outputs outside the signal");
    py::array_t<cf32> out(static_cast<py::ssize_t>(count*R));
    cf32 *o = out.mutable_data();
    const cf32 *x = x_a.data();
    const double *h = ph_a.data();
    const long long CHUNK = std::max(16LL, std::min(2048LL, (1LL << 17)/(L*R)));
    {
        py::gil_scoped_release nogil;
        parallel_for((count + CHUNK - 1)/CHUNK, threads, [&](long long blk) {
            const long long end = std::min(count, (blk + 1)*CHUNK);
            for (long long k = blk*CHUNK; k < end; k++)
                for (long long p = 0; p < R; p++) {
                    float re, im;
                    dot_back(x + start + k, h + p*L, L, &re, &im);
                    o[k*R + p] = cf32(re, im);
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
    m.def("fir_interpolate", &fir_interpolate,
          "fir_interpolate(x complex64, phases float64 (R, L), start, count, threads) -> out[k*R + p] = sum_j phases[p, j] x[start + k - j] (complex64)",
          py::arg("x"), py::arg("phases"), py::arg("start"), py::arg("count"), py::arg("threads") = 0);
}

} // namespace rpps
