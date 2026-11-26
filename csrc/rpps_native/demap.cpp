// Soft demapping of symbols to bit LLRs (rpps/math/demap.py): the noise level of each row estimated by EM over the
// constellation points, then log(P(bit=0)/P(bit=1)) per bit. Rows (frames) are independent: they run on several threads.
//
// Same arithmetic as the numpy reference. The noise comes from at most `em_symbols` of a row (every stride-th), by EM whose
// sums take the nearest point's distance out of the exponent and leave out terms under e^-40 (4e-18) of it. The LLRs share
// one exponential a point, shifted by the symbol's nearest (kept down to e^-700): every bit's sums are made of them, the
// shift cancelling in the ratio; a sum that underflows is done again with its own value's nearest point. Symbols go in
// tiles that stay in the cache, a long row in chunks over the threads.
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
#define RPPS_CLONES __attribute__((target_clones("avx2", "default")))
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

constexpr double FAR = 700.0;   // a point's term exp(-(d - dmin)/nv) is kept down to e^-700 (a normal double): one shared shift
constexpr double TINY = 1e-250; // a bit value's sum below this (its nearest point that far beyond the symbol's): done again
constexpr long long TILE = 256; // symbols a tile: its distances and terms stay in the cache
constexpr long long CHUNK = 4096; // symbols a task: a long row is split over the threads

// Distances of symbols y[0..T) from the points, dp[p*T + l], and each symbol's nearest
RPPS_CLONES
static void tile_distances(const cf64 *y, const cf64 *pts, long long T, long long P, double *dp, double *dmin) {
    for (long long l = 0; l < T; l++) dmin[l] = INFINITY;
    for (long long p = 0; p < P; p++) {
        double *d = dp + p*T;
        const double pr = pts[p].real(), pi = pts[p].imag();
        for (long long l = 0; l < T; l++) {
            const double dr = y[l].real() - pr, di = y[l].imag() - pi;
            d[l] = dr*dr + di*di;
            dmin[l] = d[l] < dmin[l] ? d[l] : dmin[l];
        }
    }
}

// Each point's term exp(-(d - dmin)/nv), shared by every bit's sums (the shift cancels in the ratio): ep[p*T + l]
RPPS_CLONES
static void tile_terms(const double *dp, const double *dmin, long long T, long long P, double inv, double *ep) {
    for (long long p = 0; p < P; p++) {
        const double *d = dp + p*T;
        double *e = ep + p*T;
        for (long long l = 0; l < T; l++) {
            const double x = (d[l] - dmin[l])*inv;
            e[l] = (x < FAR) ? expneg(x < FAR ? x : FAR) : 0.0;
        }
    }
}

// The LLRs of one tile: o[l*B + j]. Exact: log(sum over the points with bit j 0) - log(sum over those with 1) of the shared
// terms; a sum that underflows (a symbol so far beyond every point of a bit value) is the log-sum-exp done the long way.
// Max-log: the nearest point of each value
static void tile_llr(const double *dp, const double *dmin, const double *ep, const uint8_t *lab, long long T, long long P,
                     long long B, double inv, bool maxlog, double *s0, double *s1, float *o) {
    for (long long j = 0; j < B; j++) {
        if (maxlog) {
            for (long long l = 0; l < T; l++) { s0[l] = INFINITY; s1[l] = INFINITY; }
            for (long long p = 0; p < P; p++) {
                const double *d = dp + p*T;
                double *m = lab[p*B + j] ? s1 : s0;
                for (long long l = 0; l < T; l++) m[l] = d[l] < m[l] ? d[l] : m[l];
            }
            for (long long l = 0; l < T; l++) o[l*B + j] = static_cast<float>((s1[l] - s0[l])*inv);
            continue;
        }
        for (long long l = 0; l < T; l++) { s0[l] = 0.0; s1[l] = 0.0; }
        for (long long p = 0; p < P; p++) {
            const double *e = ep + p*T;
            double *s = lab[p*B + j] ? s1 : s0;
            for (long long l = 0; l < T; l++) s[l] += e[l];
        }
        for (long long l = 0; l < T; l++) {
            if (s0[l] >= TINY && s1[l] >= TINY) {
                o[l*B + j] = static_cast<float>(std::log(s0[l]) - std::log(s1[l]));
                continue;
            }
            double m0 = INFINITY, m1 = INFINITY; // the long way: each value's sum shifted by its own nearest point
            for (long long p = 0; p < P; p++) {
                const double d = dp[p*T + l];
                if (lab[p*B + j]) m1 = std::min(m1, d); else m0 = std::min(m0, d);
            }
            double a0 = 0.0, a1 = 0.0;
            for (long long p = 0; p < P; p++) {
                const double d = dp[p*T + l];
                if (lab[p*B + j]) a1 += std::exp(-(d - m1)*inv); else a0 += std::exp(-(d - m0)*inv);
            }
            o[l*B + j] = static_cast<float>((m1 - m0)*inv + std::log(a0) - std::log(a1));
        }
    }
    (void)dmin;
}

py::array_t<float> demap_llr(carray<cf64> y_a, carray<cf64> pts_a, carray<uint8_t> lab_a, carray<double> nv_a, bool maxlog,
                             int iters, double floor_var, double tol, int em_symbols, int threads) {
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
    std::vector<double> nvs(static_cast<size_t>(R));
    {
        py::gil_scoped_release nogil;
        // The noise of each row: EM over every `stride`-th symbol (at most em_symbols of them: the variance of thousands of
        // symbols is known to a few percent, and it only scales the LLRs)
        const long long stride = (em_symbols > 0 && L > em_symbols) ? (L + em_symbols - 1)/em_symbols : 1;
        const long long Ls = L ? (L + stride - 1)/stride : 0;
        parallel_for(R, threads, [&](long long r) {
            double nv;
            if (given) {
                nv = nvg[nv_a.size() == 1 ? 0 : r];
            } else {
                std::vector<cf64> ys(static_cast<size_t>(Ls));
                for (long long k = 0; k < Ls; k++) ys[k] = y_all[r*L + k*stride];
                std::vector<double> dp(static_cast<size_t>(Ls*P)), dmin(static_cast<size_t>(Ls)), s(static_cast<size_t>(Ls)),
                    t(static_cast<size_t>(Ls));
                tile_distances(ys.data(), pts, Ls, P, dp.data(), dmin.data());
                double dsum = 0.0;
                for (long long l = 0; l < Ls; l++) dsum += dmin[l];
                double var = (Ls ? dsum/Ls : 0.0)/2 + floor_var;
                for (int it = 0; it < iters && Ls; it++) {
                    const double acc = em_step(dp.data(), dmin.data(), Ls, P, 1.0/(2*var), s.data(), t.data());
                    const double nu = std::max(acc/(2.0*Ls), floor_var);
                    const bool done = std::fabs(nu - var) <= tol*var;
                    var = nu;
                    if (done) break;
                }
                nv = 2*var;
            }
            nvs[r] = std::max(nv, 2*floor_var);
        });
        // The LLRs: tasks of CHUNK symbols (a long row over several threads), tiles of TILE symbols within each
        const long long per = L ? (L + CHUNK - 1)/CHUNK : 0;
        parallel_for(R*per, threads, [&](long long task) {
            const long long r = task/per, a = (task % per)*CHUNK, z = std::min(L, a + CHUNK);
            std::vector<double> dp(static_cast<size_t>(TILE*P)), ep(static_cast<size_t>(TILE*P)), dmin(static_cast<size_t>(TILE)),
                s0(static_cast<size_t>(TILE)), s1(static_cast<size_t>(TILE));
            const double inv = 1.0/nvs[r];
            for (long long t0 = a; t0 < z; t0 += TILE) {
                const long long T = std::min(TILE, z - t0);
                tile_distances(y_all + r*L + t0, pts, T, P, dp.data(), dmin.data());
                if (!maxlog) tile_terms(dp.data(), dmin.data(), T, P, inv, ep.data());
                tile_llr(dp.data(), dmin.data(), ep.data(), lab, T, P, B, inv, maxlog, s0.data(), s1.data(), out_all + (r*L + t0)*B);
            }
        });
    }
    return out;
}

} // namespace

void register_demap(py::module_ &m) {
    m.def("demap_llr", &demap_llr,
          "demap_llr(y (R, L) complex128, points (P), labels (P, B) uint8, noise_var (R) or (1) or empty, maxlog, iters, floor, tol, em_symbols, threads) -> LLRs (R, L*B) float32",
          py::arg("y"), py::arg("points"), py::arg("labels"), py::arg("noise_var"), py::arg("maxlog"), py::arg("iters"),
          py::arg("floor"), py::arg("tol"), py::arg("em_symbols") = 4096, py::arg("threads") = 0);
}

} // namespace rpps
