// Kernels behind rpps.analysis: the loops of the blind tools that Python runs too slowly. Each mirrors a Python
// reference in its module (the reference is what the tests compare against).
#include "common.hpp"
#include "parallel.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <pybind11/stl.h>

#include <vector>

namespace rpps {

namespace {

constexpr double TWO_PI = 6.283185307179586476925286766559;

inline double wrap(double a) {
    a = std::fmod(a, TWO_PI);
    return a < 0 ? a + TWO_PI : a;
}

// rpps.analysis.cpm._mlse_python: per-survivor Viterbi over binary CPM with h = 1/2. r: n*ns samples; Q (L, ns): the
// phase a symbol j back adds at sample n; atab (2, 8): alpha for (parity, b_k + 2 b_{k-1} + 4 b_{k-2}); S state bits.
// Returns (bits (n,), the best path's metric).
py::tuple cpm_psp(carray<std::complex<double>> r, carray<double> Q, carray<int8_t> atab, int S, int parity, double k1, double k2) {
    check_shape(Q, {-1, -1}, "Q");
    check_shape(atab, {2, 8}, "atab");
    if (S < 2 || S > 16) throw std::invalid_argument("cpm_psp: S in 2..16");
    const long long L = Q.shape(0), ns = Q.shape(1);
    const long long n = static_cast<long long>(r.size()) / ns;
    const long long N = 1LL << S, half = N >> 1;
    const long long Lh = std::max(L - 1, 1LL);
    const double hpi = TWO_PI/4; // pi h, h = 1/2
    auto bits = empty<uint8_t>(static_cast<std::size_t>(n));
    double best;
    {
        py::gil_scoped_release nogil;
        const std::complex<double> *x = r.data();
        const double *q = Q.data();
        const int8_t *at = atab.data();
        std::vector<double> M(N, 0.0), phi(N, 0.0), psi(N), nu(N, 0.0), ah(N*Lh, 0.0);
        std::vector<double> M2(N), phi2(N), psi2(N), nu2(N), ah2(N*Lh);
        std::vector<int> b1(N), b2(N), b1n(N), b2n(N);
        std::vector<int32_t> prev(static_cast<std::size_t>(n*N));
        std::vector<double> cre(N*2), cim(N*2), an(N*2), mm(N*2), theta(ns);
        for (long long s = 0; s < N; s++) {
            psi[s] = TWO_PI*static_cast<double>(s)/static_cast<double>(N);
            b1[s] = s & 1;
            b2[s] = (s >> 1) & 1;
        }
        for (long long k = 0; k < n; k++) {
            const std::complex<double> *rk = x + k*ns;
            const int par = static_cast<int>((k + parity) & 1);
            for (long long s = 0; s < N; s++) {
                // the tail's phase over the symbol: symbols k-1 .. k-L+1 of this survivor
                for (long long t = 0; t < ns; t++) {
                    double th = phi[s] + psi[s];
                    for (long long j = 1; j < L; j++) th += ah[s*Lh + j - 1]*q[j*ns + t];
                    theta[t] = th;
                }
                for (int b = 0; b < 2; b++) {
                    const double a = at[par*8 + b + 2*b1[s] + 4*b2[s]];
                    double re = 0.0, im = 0.0;
                    for (long long t = 0; t < ns; t++) {
                        const double th = theta[t] + a*q[t];
                        const double c = std::cos(th), sn = std::sin(th);
                        // r e^{-j th}
                        re += rk[t].real()*c + rk[t].imag()*sn;
                        im += rk[t].imag()*c - rk[t].real()*sn;
                    }
                    an[s*2 + b] = a;
                    cre[s*2 + b] = re;
                    cim[s*2 + b] = im;
                    mm[s*2 + b] = M[s] + re;
                }
            }
            int32_t *pv = prev.data() + k*N;
            for (long long t = 0; t < N; t++) {
                const long long p0 = t >> 1, p1 = p0 | half;
                const int bb = static_cast<int>(t & 1);
                const long long w = mm[p0*2 + bb] >= mm[p1*2 + bb] ? p0 : p1;
                pv[t] = static_cast<int32_t>(w);
                const double e = std::atan2(cim[w*2 + bb], cre[w*2 + bb]);
                const double settled = L > 1 ? ah[w*Lh + L - 2] : an[w*2 + bb];
                M2[t] = mm[w*2 + bb];
                phi2[t] = wrap(phi[w] + hpi*settled);
                psi2[t] = wrap(psi[w] + k1*e + nu[w]);
                nu2[t] = nu[w] + k2*e;
                ah2[t*Lh] = an[w*2 + bb];
                for (long long j = 1; j < Lh; j++) ah2[t*Lh + j] = ah[w*Lh + j - 1];
                b2n[t] = b1[w];
                b1n[t] = bb;
            }
            M.swap(M2); phi.swap(phi2); psi.swap(psi2); nu.swap(nu2); ah.swap(ah2); b1.swap(b1n); b2.swap(b2n);
        }
        long long s = std::max_element(M.begin(), M.end()) - M.begin();
        best = M[s];
        uint8_t *o = bits.mutable_data();
        for (long long k = n - 1; k >= 0; k--) {
            o[k] = static_cast<uint8_t>(s & 1);
            s = prev[k*N + s];
        }
    }
    return py::make_tuple(bits, best);
}


// rpps.math.gf2.eliminate_packed: reduced row echelon form over GF(2) of packed rows A (R, words) in place, pivots in
// the first ncols columns (column i at word i/64, bit 63 - i%64). Method of the Four Russians: up to 32 pivots are
// found (each candidate's bit read as the block's pivots would leave it), the pivot rows reduced among themselves,
// and in a single pass (threaded) every other row takes, per group of 8 pivots, one of the 256 combinations of that
// group's rows: the matrix is streamed once per 32 pivots (it is memory-bound). The same pivot choice and
// row swaps as the Python loop, so the same matrix bit for bit. Returns the pivot columns.
std::vector<long long> gf2_eliminate(py::array_t<uint64_t, py::array::c_style> A, long long ncols, int threads) {
    check_shape(A, {-1, -1}, "A");
    const long long R = A.shape(0), Wd = A.shape(1);
    if (ncols > Wd*64) throw std::invalid_argument("gf2_eliminate: ncols over the columns");
    uint64_t *a = A.mutable_data();
    std::vector<long long> piv;
    {
        py::gil_scoped_release nogil;
        auto bit = [&](long long row, long long col) -> int {
            return static_cast<int>((a[row*Wd + (col >> 6)] >> (63 - (col & 63))) & 1ULL);
        };
        std::vector<uint64_t> tab, tmp(Wd);
        long long r = 0, col = 0;
        while (col < ncols && r < R) {
            long long pc[32], np = 0;
            const long long w0 = col >> 6; // every pivot of this block is at or after word w0
            long long c = col;
            for (; c < ncols && np < 32 && r + np < R; c++) {
                // the first row from r + np whose bit c, once the block's pivots are taken out, is set
                long long p = -1;
                for (long long i = r + np; i < R; i++) {
                    int v = bit(i, c);
                    for (long long j = 0; j < np; j++)
                        if (bit(i, pc[j])) v ^= bit(r + j, c);
                    if (v) { p = i; break; }
                }
                if (p < 0) continue;
                uint64_t *rp = a + p*Wd;
                for (long long j = 0; j < np; j++)
                    if (bit(p, pc[j])) {
                        const uint64_t *pj = a + (r + j)*Wd;
                        for (long long w = w0; w < Wd; w++) rp[w] ^= pj[w];
                    }
                if (p != r + np) {
                    uint64_t *rq = a + (r + np)*Wd;
                    std::copy(rp, rp + Wd, tmp.begin());
                    std::copy(rq, rq + Wd, rp);
                    std::copy(tmp.begin(), tmp.end(), rq);
                }
                const uint64_t *nr = a + (r + np)*Wd;
                for (long long j = 0; j < np; j++) // keep the block's pivot rows reduced among themselves
                    if (bit(r + j, c)) {
                        uint64_t *pj = a + (r + j)*Wd;
                        for (long long w = w0; w < Wd; w++) pj[w] ^= nr[w];
                    }
                pc[np++] = c;
                piv.push_back(c);
            }
            col = c;
            if (np == 0) break;
            // per group of 8 pivot rows, their 256 combinations from word w0 (Gray code: one xor each)
            const long long span = Wd - w0, ng = (np + 7)/8;
            tab.assign(static_cast<std::size_t>(ng*256*span), 0ULL);
            for (long long gr = 0; gr < ng; gr++) {
                const long long gs = std::min(8LL, np - 8*gr), T = 1LL << gs;
                uint64_t *tg = tab.data() + gr*256*span;
                for (long long g = 1; g < T; g++) {
                    const long long j = ctz64(static_cast<uint64_t>(g));
                    const uint64_t *src = tg + (g ^ (1LL << j))*span, *pr = a + (r + 8*gr + j)*Wd + w0;
                    uint64_t *dst = tg + g*span;
                    for (long long w = 0; w < span; w++) dst[w] = src[w] ^ pr[w];
                }
            }
            const long long rb = r, nb = np;
            parallel_for(R, R*span > (1 << 15) ? threads : 1, [&](long long i) {
                if (i >= rb && i < rb + nb) return;
                uint64_t *ri = a + i*Wd;
                const uint64_t *t[4];
                long long nt = 0;
                for (long long gr = 0; gr < ng; gr++) { // the row's bits at every pivot, read before any change
                    long long idx = 0;
                    for (long long j = 8*gr; j < std::min(nb, 8*gr + 8); j++)
                        idx |= static_cast<long long>((ri[pc[j] >> 6] >> (63 - (pc[j] & 63))) & 1ULL) << (j - 8*gr);
                    if (idx) t[nt++] = tab.data() + (gr*256 + idx)*span;
                }
                for (long long q = 0; q < nt; q++)
                    for (long long w = 0; w < span; w++) ri[w0 + w] ^= t[q][w];
            });
            r += np;
        }
    }
    return piv;
}


} // namespace

void register_analysis(py::module_ &m) {
    m.def("cpm_psp", &cpm_psp, "cpm_psp(r, Q (L, ns), atab (2, 8), S, parity, k1, k2) -> (bits, metric)",
          py::arg("r"), py::arg("Q"), py::arg("atab"), py::arg("S"), py::arg("parity"), py::arg("k1"), py::arg("k2"));
    m.def("gf2_eliminate", &gf2_eliminate, "gf2_eliminate(A (R, words) uint64 in place, ncols, threads) -> pivot columns",
          py::arg("A"), py::arg("ncols"), py::arg("threads") = 0);
}

} // namespace rpps
