// rpps._native: optional C++ kernels for the per-sample loops. Every kernel has a pure Python reference in rpps
#include "common.hpp"

PYBIND11_MODULE(_native, m) {
    m.doc() = "C++ kernels for RPPS per-sample loops (optional: rpps falls back to Python without them)";
    rpps::register_sync(m);
    rpps::register_filters(m);
    rpps::register_trellis(m);
    rpps::register_fec(m);
    rpps::register_spread(m);
    rpps::register_demap(m);
    rpps::register_framesync(m);
    rpps::register_firdec(m);
    rpps::register_carrier(m);
    rpps::register_resample(m);
}
