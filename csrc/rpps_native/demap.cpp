// Soft demapping of symbols to bit LLRs (rpps/math/demap.py): the noise level of each row estimated by EM over the
// constellation points, then log(P(bit=0)/P(bit=1)) per bit. Rows (frames) are independent: they run on several threads.
//
// Same arithmetic as the numpy reference, with the terms that cannot change a sum in double precision left out: an
// exponential of less than e^-40 (4e-18) of the largest term of its sum. The EM sums take the nearest point's distance out
// of the exponent; each LLR sum takes out its own bit value's nearest point, so no sum underflows and nothing needs a
// second way round.
#include "common.hpp"
#include "parallel.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace rpps {

namespace {

constexpr double CUT = 40.0; // exponents beyond this (relative to a sum's largest term) are dropped

// Loops over symbols are written to vectorise (arrays of one point's distances, no branches); on x86 the compiler also
// builds an AVX2 version and picks it when the CPU has it.
#if defined(__GNUC__) && defined(__x86_64__) && !defined(__clang__)
#define RPPS_CLONES __attribute__((target_clones("avx2,fma", "default")))
#else
#define RPPS_CLONES
#endif

// exp(-x) for x in [0, CUT]: x = n ln2 + r (n integer, |r| <= ln2/2), a degree-12 polynomial for exp(-r), scaled by 2^-n built
// from the exponent bits. Agrees with std::exp to about 1e-15 relative, and has no branch or call.
static inline double expneg(double x) {
    const double n = std::floor(x*1.4426950408889634 + 0.5);
    const double r = (x - n*6.93147180369123816490e-01) - n*1.90821492927058770002e-10; // two-part ln2
    double p = 1.0/479001600.0;
    p = p*r*(-1.0) + 1.0/39916800.0;
    p = p*r*(-1.0) + 1.0/3628800.0;
    p = p*r*(-1.0) + 1.0/362880.0;
    p = p*r*(-1.0) + 1.0/40320.0;
    p = p*r*(-1.0) + 1.0/5040.0;
    p = p*r*(-1.0) + 1.0/720.0;
    p = p*r*(-1.0) + 1.0/120.0;
    p = p*r*(-1.0) + 1.0/24.0;
    p = p*r*(-1.0) + 1.0/6.0;
    p = p*r*(-1.0) + 0.5;
    p = p*r*(-1.0) + 1.0;
    p = p*r*(-1.0) + 1.0;
    union { double d; uint64_t u; } scale;
    scale.u = static_cast<uint64_t>(1023 - static_cast<int64_t>(n)) << 52; // 2^-n
    return p*scale.d;
}

// One EM step over symbols l = 0..L-1, distances dp[p*L + l]: sum_l (sum_p w d)/(sum_p w), w = exp(-(d - dmin)/(2 var)), terms
// beyond CUT dropped
RPPS_CLONES
static double em_step(const double *dp, const double *dmin, long long L, long long P, double inv, double *s, double *t) {
    for (long long l = 0; l < L; l++) { s[l] = 0.0; t[l] = 0.0; }
    for (long long p = 0; p < P; p++) {
        const double *d = dp + p*L;
        for (long long l = 0; l < L; l++) {
            const double e = (d[l] - dmin[l])*inv;
            const double w = (e < CUT) ? expneg(e < CUT ? e : CUT) : 0.0;
            s[l] += w;
            t[l] += w*d[l];
        }
    }
    double acc = 0.0;
    for (long long l = 0; l < L; l++) acc += t[l]/s[l];
    return acc;
}

// LLR of bit j for every symbol: (m1 - m0)/nv + log(s0) - log(s1), the sums over the points with that bit value 0 / 1, each
// shifted by the value's nearest point (m0, m1)
RPPS_CLONES
static void bit_llr(const double *dp, const uint8_t *lab, long long L, long long P, long long B, long long j, double inv,
                    bool maxlog, double *m0, double *m1, double *s0, double *s1, float *o) {
    for (long long l = 0; l < L; l++) { m0[l] = INFINITY; m1[l] = INFINITY; }
    for (long long p = 0; p < P; p++) {
        const double *d = dp + p*L;
        double *m = lab[p*B + j] ? m1 : m0;
        for (long long l = 0; l < L; l++) m[l] = d[l] < m[l] ? d[l] : m[l];
    }
    if (maxlog) {
        for (long long l = 0; l < L; l++) o[l*B + j] = static_cast<float>((m1[l] - m0[l])*inv);
        return;
    }
    for (long long l = 0; l < L; l++) { s0[l] = 0.0; s1[l] = 0.0; }
    for (long long p = 0; p < P; p++) {
        const double *d = dp + p*L;
        double *s = lab[p*B + j] ? s1 : s0;
        const double *m = lab[p*B + j] ? m1 : m0;
        for (long long l = 0; l < L; l++) {
            const double e = (d[l] - m[l])*inv;
            s[l] += (e < CUT) ? expneg(e < CUT ? e : CUT) : 0.0;
        }
    }
    for (long long l = 0; l < L; l++)
        o[l*B + j] = static_cast<float>((m1[l] - m0[l])*inv + std::log(s0[l]) - std::log(s1[l]));
}

py::array_t<float> demap_llr(carray<cf64> y_a, carray<cf64> pts_a, carray<uint8_t> lab_a, carray<double> nv_a, bool maxlog,
                             int iters, double floor_var, double tol, int threads) {
    check_shape(y_a, {-1, -1}, "y");
    check_shape(lab_a, {-1, -1}, "labels");
    const long long R = y_a.shape(0), L = y_a.shape(1), P = pts_a.size(), B = lab_a.shape(1);
    if (lab_a.shape(0) != P) throw std::invalid_argument("labels: one row per point");
    const bool given = nv_a.size() > 0;
    if (given && nv_a.size() != R && nv_a.size() != 1) throw std::invalid_argument("noise_var: one per row, or one");
    const cf64 *y_all = y_a.data(), *pts = pts_a.data();
    const uint8_t *lab = lab_a.data();
    const double *nvg = nv_a.data();
    py::array_t<float> out({R, L*B});
    float *out_all = out.mutable_data();
    {
        py::gil_scoped_release nogil;
        parallel_for(R, threads, [&](long long r) {
            const cf64 *y = y_all + r*L;
            std::vector<double> dp(static_cast<size_t>(L*P)), dmin(static_cast<size_t>(L), INFINITY), s(static_cast<size_t>(L)),
                t(static_cast<size_t>(L)), a0(static_cast<size_t>(L)), a1(static_cast<size_t>(L));
            for (long long p = 0; p < P; p++) { // distances, one point at a time: dp[p*L + l]
                double *d = &dp[p*L];
                const double pr = pts[p].real(), pi = pts[p].imag();
                for (long long l = 0; l < L; l++) {
                    const double dr = y[l].real() - pr, di = y[l].imag() - pi;
                    d[l] = dr*dr + di*di;
                    dmin[l] = d[l] < dmin[l] ? d[l] : dmin[l];
                }
            }
            double nv;
            if (given) {
                nv = nvg[nv_a.size() == 1 ? 0 : r];
            } else {
                double dsum = 0.0;
                for (long long l = 0; l < L; l++) dsum += dmin[l];
                double var = (L ? dsum/L : 0.0)/2 + floor_var;
                for (int it = 0; it < iters && L; it++) {
                    const double acc = em_step(dp.data(), dmin.data(), L, P, 1.0/(2*var), s.data(), t.data());
                    const double nu = std::max(acc/(2.0*L), floor_var);
                    const bool done = std::fabs(nu - var) <= tol*var;
                    var = nu;
                    if (done) break;
                }
                nv = 2*var;
            }
            nv = std::max(nv, 2*floor_var);
            float *o = out_all + r*L*B;
            for (long long j = 0; j < B; j++)
                bit_llr(dp.data(), lab, L, P, B, j, 1.0/nv, maxlog, dmin.data(), t.data(), a0.data(), a1.data(), o);
        });
    }
    return out;
}

} // namespace

void register_demap(py::module_ &m) {
    m.def("demap_llr", &demap_llr,
          "demap_llr(y (R, L) complex128, points (P), labels (P, B) uint8, noise_var (R) or (1) or empty, maxlog, iters, floor, tol, threads) -> LLRs (R, L*B) float32",
          py::arg("y"), py::arg("points"), py::arg("labels"), py::arg("noise_var"), py::arg("maxlog"), py::arg("iters"),
          py::arg("floor"), py::arg("tol"), py::arg("threads") = 0);
}

} // namespace rpps
