// Block carrier estimation helpers (rpps/sync/phase/mpower.py) and the frame sync's preamble metric (rpps/sync/frame.py).
// The windows are short, so what costs in numpy is the dozen array operations per window; here each window is one call, and
// the phase line is a rotating phasor (one exact start per 64 samples, a multiplication per sample between), not an
// exponential per sample.
#include "common.hpp"
#include "parallel.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace rpps {

namespace {

constexpr long long RESTART = 64; // samples between exact phasor starts

// e^{-j (phi0 + rate i)}, i = 0 .. n-1, handed to f(i, wr, wi). Plain real arithmetic throughout: std::complex products and
// std::abs are far slower (NaN handling, hypot).
template <typename F>
static inline void line(double phi0, double rate, long long n, F f) {
    const double sr = std::cos(rate), si = -std::sin(rate);
    for (long long a = 0; a < n; a += RESTART) {
        const double ph = phi0 + rate*static_cast<double>(a);
        double wr = std::cos(ph), wi = -std::sin(ph);
        const long long end = std::min(n, a + RESTART);
        for (long long i = a; i < end; i++) {
            f(i, wr, wi);
            const double t = wr*sr - wi*si;
            wi = wr*si + wi*sr;
            wr = t;
        }
    }
}

// (sum_i z_i, sum_i |z_i|) of z_i = unit(y_i e^{-j(phi0 + rate i)})^M: what a window's carrier measurement is made of.
// unit(y) = y / max(|y|, 1e-30), so a zero stays zero; the power is by squaring
py::tuple mpower_sum(carray<cf64> y_a, double phi0, double rate, int M) {
    const long long n = y_a.size();
    const cf64 *y = y_a.data();
    double sr = 0.0, si = 0.0, mag = 0.0;
    line(phi0, rate, n, [&](long long i, double wr, double wi) {
        const double yr = y[i].real()*wr - y[i].imag()*wi, yi = y[i].real()*wi + y[i].imag()*wr;
        const double a = std::sqrt(yr*yr + yi*yi), inv = 1.0/std::max(a, 1e-30);
        double ur = yr*inv, ui = yi*inv, rr = 1.0, ri = 0.0, um = a*inv, rm = 1.0; // um: |u|
        for (int m = M; m; m >>= 1) {
            if (m & 1) { const double t = rr*ur - ri*ui; ri = rr*ui + ri*ur; rr = t; rm *= um; }
            const double t = ur*ur - ui*ui; ui = 2*ur*ui; ur = t; um *= um;
        }
        sr += rr; si += ri; mag += rm;
    });
    return py::make_tuple(cf64(sr, si), mag);
}

// y turned back by the phase line, as single precision
py::array_t<cf32> rotate_line(carray<cf64> y_a, double phi0, double rate) {
    const long long n = y_a.size();
    py::array_t<cf32> out(static_cast<py::ssize_t>(n));
    cf32 *o = out.mutable_data();
    const cf64 *y = y_a.data();
    line(phi0, rate, n, [&](long long i, double wr, double wi) {
        o[i] = cf32(static_cast<float>(y[i].real()*wr - y[i].imag()*wi), static_cast<float>(y[i].real()*wi + y[i].imag()*wr));
    });
    return out;
}

// The preamble metric at every position: c[j] = sum_k y[j+k] conj(pre[k]) and m[j] = |c[j]| / sqrt(max(E[j], 1e-30) Ep), E the
// energy of the window y[j .. j+L). Direct, on several threads: a short preamble costs L multiplications a position.
py::tuple preamble_metric(carray<cf64> y_a, carray<cf64> pre_a, double ep, int threads) {
    const long long S = y_a.size(), L = pre_a.size(), nw = S - L + 1;
    if (L < 1) throw std::invalid_argument("preamble_metric: empty preamble");
    const long long n = std::max(nw, 0LL);
    py::array_t<cf64> c(static_cast<py::ssize_t>(n));
    py::array_t<double> m(static_cast<py::ssize_t>(n));
    cf64 *co = c.mutable_data();
    double *mo = m.mutable_data();
    const cf64 *y = y_a.data(), *pre = pre_a.data();
    constexpr long long CHUNK = 4096;
    {
        py::gil_scoped_release nogil;
        parallel_for((n + CHUNK - 1)/CHUNK, threads, [&](long long blk) {
            const long long a = blk*CHUNK, end = std::min(n, a + CHUNK);
            double e = 0.0; // the window energy, slid along; started afresh every chunk so rounding does not build up
            for (long long k = 0; k < L; k++) e += std::norm(y[a + k]);
            for (long long j = a; j < end; j++) {
                double cr = 0.0, ci = 0.0;
                for (long long k = 0; k < L; k++) { // y[j+k] * conj(pre[k])
                    const double yr = y[j + k].real(), yi = y[j + k].imag(), pr = pre[k].real(), pi = pre[k].imag();
                    cr += yr*pr + yi*pi;
                    ci += yi*pr - yr*pi;
                }
                co[j] = cf64(cr, ci);
                mo[j] = std::sqrt(cr*cr + ci*ci)/std::sqrt(std::max(e, 1e-30)*ep);
                if (j + 1 < end) e += std::norm(y[j + L]) - std::norm(y[j]);
                e = std::max(e, 0.0);
            }
        });
    }
    return py::make_tuple(c, m);
}

} // namespace

void register_carrier(py::module_ &m) {
    m.def("mpower_sum", &mpower_sum, "mpower_sum(y complex128, phi0, rate, M) -> (sum z, sum |z|), z = unit(y e^{-j(phi0 + rate i)})^M",
          py::arg("y"), py::arg("phi0"), py::arg("rate"), py::arg("order"));
    m.def("rotate_line", &rotate_line, "rotate_line(y complex128, phi0, rate) -> y e^{-j(phi0 + rate i)} (complex64)",
          py::arg("y"), py::arg("phi0"), py::arg("rate"));
    m.def("preamble_metric", &preamble_metric, "preamble_metric(y complex128, preamble complex128, energy, threads) -> (c, m): the correlation and the normalised metric of every position",
          py::arg("y"), py::arg("preamble"), py::arg("energy"), py::arg("threads") = 0);
}

} // namespace rpps
