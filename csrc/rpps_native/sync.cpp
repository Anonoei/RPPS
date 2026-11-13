// Synchronization loops: Costas, AGC, Mueller & Muller / Gardner timing recovery
//
// Each mirrors its Python stage (rpps/sync/costas.py, agc.py, time/mm.py) operation for operation, in the precision
// the Python code ends up computing in, so the two agree to rounding.
#include "common.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace rpps {

namespace {

constexpr double TWO_PI = 6.283185307179586;

// ---------------------------------------------------------------- Costas
// order 2: BPSK, 4: QPSK (points at 45 degrees), other M >= 3: M-PSK with a point at `offset`. Double arithmetic
// (numpy upcasts the symbol times exp(-j*pha) to complex128), the output keeps the sample precision.
template <typename T>
py::tuple costas(carray<std::complex<T>> x, double pha, double freq, double alpha, double beta, int order, double offset) {
    const std::size_t n = static_cast<std::size_t>(x.size());
    auto out = empty<std::complex<T>>(n);
    const std::complex<T> *in = x.data();
    std::complex<T> *o = out.mutable_data();
    const double step = TWO_PI / order;
    {
        py::gil_scoped_release nogil;
        for (std::size_t i = 0; i < n; i++) {
            const double a = in[i].real(), b = in[i].imag();
            const double c = std::cos(pha), d = -std::sin(pha);
            const double re = a*c - b*d, im = a*d + b*c;
            double err;
            if (order == 2) {
                err = re*im / (re*re + im*im + 1e-12);
            } else if (order == 4) {
                const double sa = re > 0 ? 1.0 : -1.0, sb = im > 0 ? 1.0 : -1.0;
                err = (sa*im - sb*re) / (std::hypot(re, im) + 1e-12);
            } else {
                const double ang = std::atan2(im, re) - offset;
                err = ang - std::nearbyint(ang/step)*step;
            }
            freq += beta*err;
            pha += freq + alpha*err;
            while (pha >= TWO_PI) pha -= TWO_PI;
            while (pha < 0) pha += TWO_PI;
            o[i] = std::complex<T>(static_cast<T>(re), static_cast<T>(im));
        }
    }
    return py::make_tuple(out, pha, freq);
}

// ---------------------------------------------------------------- AGC
// z = gain*|x|; linear: gain += mu*(target - z), log: gain *= exp(mu*(log(target) - log(z))), clipped to [lo, hi]
template <typename S>
py::tuple agc(carray<S> x, double gain, double mu, double target, double lo, double hi, bool log) {
    const std::size_t n = static_cast<std::size_t>(x.size());
    auto out = empty<S>(n);
    const S *in = x.data();
    S *o = out.mutable_data();
    {
        py::gil_scoped_release nogil;
        const double ltarget = std::log(target);
        for (std::size_t i = 0; i < n; i++) {
            const double a = static_cast<double>(std::abs(in[i]));
            o[i] = in[i] * static_cast<decltype(std::abs(in[i]))>(gain);
            if (log)
                gain = gain*std::exp(mu*(ltarget - std::log(gain*a + 1e-30)));
            else
                gain = gain + mu*(target - gain*a);
            gain = std::min(std::max(gain, lo), hi);
        }
    }
    return py::make_tuple(out, gain);
}

// ---------------------------------------------------------------- Mueller & Muller / Gardner / early-late
// The symbol walk of MM.step. MM interpolates the whole block (zero stuffing by `ratio` and np.convolve(.., 'same')
// with the sinc pulse), then reads ~3 points per symbol: here only those points are computed, ~L/ratio taps each.
// Numbers as in the Python stage: interpolated points and loop state in double, the M&M history (outs, rail) in
// complex64 with its error in float. outs/rail: 3 values each, newest first.
enum Ted { TED_MM = 0, TED_GARDNER = 1 };

inline double pw2(cf64 v) {
    const double a = std::hypot(v.real(), v.imag());
    return a*a;
}

py::tuple mm(carray<cf32> x_a, carray<double> pulse_a, long long i_in, double est, long long count, double pw, double pw_half,
             carray<cf32> outs_a, carray<cf32> rail_a, int ratio, int sps, double alpha, int guard, int ted, bool biphase) {
    if (outs_a.size() != 3 || rail_a.size() != 3)
        throw std::invalid_argument("outs and rail hold 3 values");
    const cf32 *x = x_a.data();
    const double *pulse = pulse_a.data();
    const long long n_in = static_cast<long long>(x_a.size()), L = static_cast<long long>(pulse_a.size());
    const long long len = n_in*ratio, centre = (L - 1)/2;
    if (len < L)
        throw std::invalid_argument("mm: block shorter than the interpolation pulse");
    cf32 outs[3] = {outs_a.data()[0], outs_a.data()[1], outs_a.data()[2]};
    cf32 rail[3] = {rail_a.data()[0], rail_a.data()[1], rail_a.data()[2]};
    const int half = sps/2;
    std::vector<cf32> out;
    out.reserve(static_cast<std::size_t>(std::max<long long>(0, 2*(n_in - i_in)/std::max(1, sps - half) + 4)));
    bool bad_index = false;
    {
        py::gil_scoped_release nogil;
        auto at = [&](long long j) -> cf64 { // interp[j], numpy indexing: negative counts from the end
            if (j < 0) j += len;
            if (j < 0 || j >= len) { bad_index = true; return cf64(0.0, 0.0); }
            const long long k0 = j + centre; // interp[j] = sum_m x[m] pulse[k0 - m*ratio]
            long long m_lo = k0 - (L - 1) <= 0 ? 0 : (k0 - (L - 1) + ratio - 1)/ratio;
            long long m_hi = std::min(n_in - 1, k0/ratio);
            double re = 0.0, im = 0.0;
            for (long long m = m_lo; m <= m_hi; m++) {
                const double h = pulse[k0 - m*ratio];
                re += x[m].real()*h; im += x[m].imag()*h;
            }
            return cf64(re, im);
        };
        auto clock = [&](cf32 v, cf32 r) {
            outs[2] = outs[1]; outs[1] = outs[0]; outs[0] = v;
            rail[2] = rail[1]; rail[1] = rail[0]; rail[0] = r;
        };
        while (i_in + guard < n_in && !bad_index) {
            const long long opt = i_in*ratio + static_cast<long long>(est*ratio);
            const cf64 val = at(opt);
            const double p = pw2(val);
            pw = count == 0 ? p : pw + 0.02*(p - pw);
            if (biphase) {
                const double ph = pw2(at(opt - (static_cast<long long>(sps)*ratio)/2));
                pw_half = count == 0 ? ph : pw_half + 0.02*(ph - pw_half);
                if (count > 100 && pw_half > 1.3*pw) { // locked to the wrong half-symbol: hop to the stronger peak
                    i_in -= half;
                    pw = pw_half = std::max(pw, pw_half);
                    continue;
                }
            }
            const cf32 v32(static_cast<float>(val.real()), static_cast<float>(val.imag()));
            out.push_back(v32);
            double err;
            if (biphase) {
                const long long d = static_cast<long long>(ratio)*std::max(1, sps/8);
                const double early = pw2(at(opt - d)), late = pw2(at(opt + d));
                err = (late - early) / (early + late + 1e-12);
            } else if (ted == TED_GARDNER) {
                const cf64 mid = at(opt - (static_cast<long long>(sps)*ratio)/2);
                clock(v32, cf32(0.0f, 0.0f));
                const cf32 dlt = outs[0] - outs[1];
                err = -(static_cast<double>(dlt.real())*mid.real() + static_cast<double>(dlt.imag())*mid.imag()) / (pw + 1e-12);
            } else {
                clock(v32, cf32(val.real() > 0 ? 1.0f : 0.0f, val.imag() > 0 ? 1.0f : 0.0f));
                const cf32 xr = rail[0] - rail[2], yo = outs[0] - outs[2];
                const float xe = xr.real()*outs[1].real() + xr.imag()*outs[1].imag();
                const float ye = yo.real()*rail[1].real() + yo.imag()*rail[1].imag();
                err = static_cast<double>(ye - xe) / (std::sqrt(pw) + 1e-12);
            }
            est += alpha*err;
            count += 1;
            const double fl = std::floor(est);
            const long long step = static_cast<long long>(std::min(std::max(fl, static_cast<double>(-half)), static_cast<double>(half)));
            est -= static_cast<double>(step);
            i_in += sps + step;
        }
    }
    if (bad_index)
        throw py::index_error("mm: interpolation index out of range");
    auto o = empty<cf32>(out.size());
    std::copy(out.begin(), out.end(), o.mutable_data());
    auto so = empty<cf32>(3), sr = empty<cf32>(3);
    std::copy(outs, outs + 3, so.mutable_data());
    std::copy(rail, rail + 3, sr.mutable_data());
    return py::make_tuple(o, i_in, est, count, pw, pw_half, so, sr);
}

// ---------------------------------------------------------------- Adaptive equalizer (rpps/equalize.py): CMA and LMS
// One output per `sps` inputs: y = w . (buf[n], buf[n-1], ..., buf[n-N+1]), then w += mu*err*conj(x). complex64 throughout,
// as numpy computes it. mode 0: CMA, err = -y(|y|^2 - R). mode 1: LMS, err = d - y with d the training symbol while
// there is one, else the nearest of `points` (none: no update).
py::tuple equalize(carray<cf32> buf_a, carray<cf32> w_a, long long n, int sps, double mu_d, int mode, double R_d,
                   carray<cf32> points_a, bool have_points, carray<cf32> training_a, bool have_training, long long count) {
    const cf32 *buf = buf_a.data(), *points = points_a.data(), *training = training_a.data();
    const long long len = static_cast<long long>(buf_a.size()), N = static_cast<long long>(w_a.size());
    const long long np_ = static_cast<long long>(points_a.size()), nt = static_cast<long long>(training_a.size());
    auto wo = empty<cf32>(static_cast<std::size_t>(N));
    cf32 *w = wo.mutable_data();
    std::copy(w_a.data(), w_a.data() + N, w);
    const float mu = static_cast<float>(mu_d), R = static_cast<float>(R_d);
    std::vector<cf32> out;
    out.reserve(static_cast<std::size_t>(std::max<long long>(0, (len - n)/std::max(1, sps) + 1)));
    {
        py::gil_scoped_release nogil;
        while (n < len) {
            float yr = 0.0f, yi = 0.0f;
            for (long long k = 0; k < N; k++) {
                const cf32 a = w[k], b = buf[n - k];
                yr += a.real()*b.real() - a.imag()*b.imag();
                yi += a.real()*b.imag() + a.imag()*b.real();
            }
            const cf32 y(yr, yi);
            cf32 err(0.0f, 0.0f);
            if (mode == 0) {
                const float a = std::hypot(y.real(), y.imag());
                const float g = a*a - R;
                err = cf32(-y.real()*g, -y.imag()*g);
            } else if (have_training && count < nt) {
                err = training[count] - y;
            } else if (have_points && np_ > 0) {
                long long best = 0; float bd = std::numeric_limits<float>::infinity();
                for (long long j = 0; j < np_; j++) {
                    const cf32 d = points[j] - y;
                    const float dist = std::hypot(d.real(), d.imag());
                    if (dist < bd) { bd = dist; best = j; }
                }
                err = points[best] - y;
            }
            const cf32 me(mu*err.real(), mu*err.imag());
            for (long long k = 0; k < N; k++) {
                const cf32 x = buf[n - k];
                // (mu*err) * conj(x)
                w[k] += cf32(me.real()*x.real() + me.imag()*x.imag(), me.imag()*x.real() - me.real()*x.imag());
            }
            out.push_back(y);
            n += sps;
            count += 1;
        }
    }
    auto o = empty<cf32>(out.size());
    std::copy(out.begin(), out.end(), o.mutable_data());
    return py::make_tuple(o, wo, n, count);
}

} // namespace

void register_sync(py::module_ &m) {
    const char *costas_doc = "costas(x, pha, freq, alpha, beta, order, offset) -> (y, pha, freq)";
    m.def("costas", &costas<float>, costas_doc, py::arg("x").noconvert(), py::arg("pha"), py::arg("freq"), py::arg("alpha"), py::arg("beta"), py::arg("order"), py::arg("offset"));
    m.def("costas", &costas<double>, costas_doc, py::arg("x"), py::arg("pha"), py::arg("freq"), py::arg("alpha"), py::arg("beta"), py::arg("order"), py::arg("offset"));

    const char *agc_doc = "agc(x, gain, mu, target, lo, hi, log) -> (y, gain)";
    m.def("agc", &agc<cf32>, agc_doc, py::arg("x").noconvert(), py::arg("gain"), py::arg("mu"), py::arg("target"), py::arg("lo"), py::arg("hi"), py::arg("log"));
    m.def("agc", &agc<float>, agc_doc, py::arg("x").noconvert(), py::arg("gain"), py::arg("mu"), py::arg("target"), py::arg("lo"), py::arg("hi"), py::arg("log"));
    m.def("agc", &agc<double>, agc_doc, py::arg("x").noconvert(), py::arg("gain"), py::arg("mu"), py::arg("target"), py::arg("lo"), py::arg("hi"), py::arg("log"));
    m.def("agc", &agc<cf64>, agc_doc, py::arg("x"), py::arg("gain"), py::arg("mu"), py::arg("target"), py::arg("lo"), py::arg("hi"), py::arg("log"));

    m.def("equalize", &equalize, "equalize(buf, w, n, sps, mu, mode, R, points, have_points, training, have_training, count) -> (y, w, n, count)",
          py::arg("buf"), py::arg("w"), py::arg("n"), py::arg("sps"), py::arg("mu"), py::arg("mode"), py::arg("R"),
          py::arg("points"), py::arg("have_points"), py::arg("training"), py::arg("have_training"), py::arg("count"));
    m.def("mm", &mm, "mm(x, pulse, i_in, est, count, pw, pw_half, outs, rail, ratio, sps, alpha, guard, ted, biphase) -> (y, i_in, est, count, pw, pw_half, outs, rail)",
          py::arg("x"), py::arg("pulse"), py::arg("i_in"), py::arg("est"), py::arg("count"), py::arg("pw"), py::arg("pw_half"),
          py::arg("outs"), py::arg("rail"), py::arg("ratio"), py::arg("sps"), py::arg("alpha"), py::arg("guard"), py::arg("ted"), py::arg("biphase"));
}

} // namespace rpps
