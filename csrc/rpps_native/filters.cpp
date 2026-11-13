// Recursive filters: the IIR section cascade (rpps/filters/iir.py) and the one-pole loops of rpps/demod.py
#include "common.hpp"

#include <cmath>

namespace rpps {

namespace {

// ---------------------------------------------------------------- IIR: second-order sections, transposed direct form II
// x: (n, c) columns filtered alike (c = 1 real, 2 real and imaginary); sos: (S, 6) rows [b0 b1 b2 1 a1 a2];
// z: (S, 2, c) state, the same state space IIR keeps: y = b0 x + z1, z1 <- -a1 z1 + z2 + (b1 - a1 b0) x, z2 <- -a2 z1 + (b2 - a2 b0) x
py::tuple sos_filter(carray<double> x_a, carray<double> sos_a, carray<double> z_a) {
    check_shape(x_a, {-1, -1}, "x");
    const py::ssize_t n = x_a.shape(0), c = x_a.shape(1);
    check_shape(sos_a, {-1, 6}, "sos");
    const py::ssize_t S = sos_a.shape(0);
    check_shape(z_a, {S, 2, c}, "z");
    py::array_t<double> y_a({n, c});
    py::array_t<double> zo_a({S, static_cast<py::ssize_t>(2), c});
    const double *x = x_a.data(), *sos = sos_a.data();
    double *y = y_a.mutable_data(), *z = zo_a.mutable_data();
    std::copy(z_a.data(), z_a.data() + S*2*c, z);
    {
        py::gil_scoped_release nogil;
        for (py::ssize_t col = 0; col < c; col++) {
            for (py::ssize_t i = 0; i < n; i++) {
                double v = x[i*c + col];
                for (py::ssize_t s = 0; s < S; s++) {
                    const double *r = sos + s*6;
                    double &z1 = z[(s*2 + 0)*c + col], &z2 = z[(s*2 + 1)*c + col];
                    const double out = r[0]*v + z1;
                    const double n1 = -r[4]*z1 + z2 + (r[1] - r[4]*r[0])*v;
                    z2 = -r[5]*z1 + (r[2] - r[5]*r[0])*v;
                    z1 = n1;
                    v = out;
                }
                y[i*c + col] = v;
            }
        }
    }
    return py::make_tuple(y_a, zo_a);
}

// ---------------------------------------------------------------- demod.py loops
// AMDemod: out = |x| - mean, the mean following |x| by alpha per sample (starting at the first sample when `have` is false)
py::tuple am_demod(carray<float> env_a, double mean, bool have, double alpha) {
    const std::size_t n = static_cast<std::size_t>(env_a.size());
    auto out = empty<float>(n);
    const float *env = env_a.data();
    float *o = out.mutable_data();
    {
        py::gil_scoped_release nogil;
        for (std::size_t i = 0; i < n; i++) {
            const double v = env[i];
            mean = have ? mean + alpha*(v - mean) : v;
            have = true;
            o[i] = static_cast<float>(v - mean);
        }
    }
    return py::make_tuple(out, mean, have);
}

// Squelch: passes samples while the smoothed power is above `hi` (opening) / `lo` (closing), zeros otherwise
template <typename S>
py::tuple squelch(carray<S> x_a, double level, bool have, bool open, double lo, double hi, double alpha) {
    const std::size_t n = static_cast<std::size_t>(x_a.size());
    auto out = empty<S>(n);
    const S *x = x_a.data();
    S *o = out.mutable_data();
    {
        py::gil_scoped_release nogil;
        for (std::size_t i = 0; i < n; i++) {
            const double a = static_cast<double>(std::abs(x[i]));
            const double p = a*a;
            level = have ? level + alpha*(p - level) : p;
            have = true;
            open = level > (open ? lo : hi);
            o[i] = open ? x[i] : S(0);
        }
    }
    return py::make_tuple(out, level, have, open);
}

// DCBlock: y = x - x[n-1] + r*y[n-1]
template <typename S, typename A>
py::tuple dc_block(carray<S> x_a, A x1, A y1, double r) {
    const std::size_t n = static_cast<std::size_t>(x_a.size());
    auto out = empty<S>(n);
    const S *x = x_a.data();
    S *o = out.mutable_data();
    {
        py::gil_scoped_release nogil;
        for (std::size_t i = 0; i < n; i++) {
            const A v = static_cast<A>(x[i]);
            y1 = v - x1 + r*y1;
            x1 = v;
            o[i] = static_cast<S>(y1);
        }
    }
    return py::make_tuple(out, x1, y1);
}

// Deemphasis (and any one-pole low pass): y += a*(x - y)
template <typename S, typename A>
py::tuple one_pole(carray<S> x_a, A y, double a) {
    const std::size_t n = static_cast<std::size_t>(x_a.size());
    auto out = empty<S>(n);
    const S *x = x_a.data();
    S *o = out.mutable_data();
    {
        py::gil_scoped_release nogil;
        for (std::size_t i = 0; i < n; i++) {
            y += a*(static_cast<A>(x[i]) - y);
            o[i] = static_cast<S>(y);
        }
    }
    return py::make_tuple(out, y);
}

} // namespace

void register_filters(py::module_ &m) {
    m.def("sos_filter", &sos_filter, "sos_filter(x (n, c), sos (S, 6), z (S, 2, c)) -> (y, z)", py::arg("x"), py::arg("sos"), py::arg("z"));
    m.def("am_demod", &am_demod, "am_demod(env, mean, have, alpha) -> (out, mean, have)", py::arg("env"), py::arg("mean"), py::arg("have"), py::arg("alpha"));

#define RPPS_DEF_SQUELCH(S) m.def("squelch", &squelch<S>, "squelch(x, level, have, open, lo, hi, alpha) -> (y, level, have, open)", \
        py::arg("x").noconvert(), py::arg("level"), py::arg("have"), py::arg("open"), py::arg("lo"), py::arg("hi"), py::arg("alpha"))
    RPPS_DEF_SQUELCH(float); RPPS_DEF_SQUELCH(double); RPPS_DEF_SQUELCH(cf32); RPPS_DEF_SQUELCH(cf64);
#undef RPPS_DEF_SQUELCH

#define RPPS_DEF_DC(S, A) m.def("dc_block", &dc_block<S, A>, "dc_block(x, x1, y1, r) -> (y, x1, y1)", py::arg("x").noconvert(), py::arg("x1"), py::arg("y1"), py::arg("r"))
    RPPS_DEF_DC(float, double); RPPS_DEF_DC(double, double); RPPS_DEF_DC(cf32, cf64); RPPS_DEF_DC(cf64, cf64);
#undef RPPS_DEF_DC

#define RPPS_DEF_POLE(S, A) m.def("one_pole", &one_pole<S, A>, "one_pole(x, y, a) -> (out, y)", py::arg("x").noconvert(), py::arg("y"), py::arg("a"))
    RPPS_DEF_POLE(float, double); RPPS_DEF_POLE(double, double); RPPS_DEF_POLE(cf32, cf64); RPPS_DEF_POLE(cf64, cf64);
#undef RPPS_DEF_POLE
}

} // namespace rpps
