// rpps._native: optional C++ kernels for the per-sample loops. Every kernel has a pure Python reference in rpps
#include "common.hpp"

PYBIND11_MODULE(_native, m) {
    m.doc() = "C++ kernels for RPPS per-sample loops (optional: rpps falls back to Python without them)";
#ifdef RPPS_SRC_HASH // rpps._srchash of csrc/rpps_native when built (setup.py), a bare token: quotes in -D differ by compiler
#define RPPS_STR2(x) #x
#define RPPS_STR(x) RPPS_STR2(x)
    m.attr("SRC_HASH") = RPPS_STR(RPPS_SRC_HASH);
#endif
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
    rpps::register_analysis(m);
    rpps::register_polar(m);
}
