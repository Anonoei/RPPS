// Block FEC decoders: LDPC min-sum (rpps/coding/types/ldpc.py) and the algebraic decoding of cyclic codes over
// GF(2^m) (rpps/math/cyclic.py: Reed-Solomon with erasures, binary BCH). Codewords are independent: batches run on
// several threads.
#include "common.hpp"
#include "parallel.hpp"

#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace rpps {

namespace {

constexpr double INF = std::numeric_limits<double>::infinity();

// Layered normalized min-sum (rpps/code/types/ldpc.py min_sum): the checks in `rows` order (a colouring's layers one after
// the other: within a layer no two checks share a variable, so going row by row gives what the numpy decoder's layer at a
// time does). For each check: q = total - r (the variable's value without this check), the sign product and the two
// smallest |q|, r = alpha * sign * the smallest of the others, total = q + r. Each codeword stops when every check holds.
py::tuple ldpc_minsum(carray<double> llr, carray<int64_t> var_a, carray<int64_t> cstart_a, carray<int64_t> deg_a,
                      carray<int64_t> rows_a, int iters, double alpha, int threads) {
    check_shape(llr, {-1, -1}, "llr");
    const long long B = llr.shape(0), N = llr.shape(1), E = var_a.size(), M = cstart_a.size(), R = rows_a.size();
    check_shape(deg_a, {M}, "deg");
    const int64_t *var = var_a.data(), *cstart = cstart_a.data(), *deg = deg_a.data(), *rows = rows_a.data();
    for (long long e = 0; e < E; e++)
        if (var[e] < 0 || var[e] >= N) throw std::invalid_argument("ldpc: edge tables out of range");
    for (long long c = 0; c < M; c++)
        if (deg[c] < 1 || cstart[c] < 0 || cstart[c] + deg[c] > E) throw std::invalid_argument("ldpc: check tables out of range");
    for (long long k = 0; k < R; k++)
        if (rows[k] < 0 || rows[k] >= M) throw std::invalid_argument("ldpc: row order out of range");
    py::array_t<bool> hard({B, N});
    py::array_t<bool> conv(static_cast<py::ssize_t>(B));
    const double *ch_all = llr.data();
    bool *hd = hard.mutable_data(), *cv = conv.mutable_data();
    {
        py::gil_scoped_release nogil;
        parallel_for(B, threads, [&](long long b) {
            const double *ch = ch_all + b*N;
            std::vector<double> total(ch, ch + N), r(E, 0.0), q;
            bool done = false;
            for (int it = 0; it < iters && !done; it++) {
                for (long long k = 0; k < R; k++) {
                    const long long c = rows[k], e0 = cstart[c], d = deg[c];
                    q.resize(static_cast<size_t>(d));
                    int negs = 0; long long i1 = 0; double min1 = INF;
                    for (long long j = 0; j < d; j++) {
                        const double v = total[var[e0 + j]] - r[e0 + j];
                        q[j] = v;
                        if (v < 0) negs++;
                        const double a = std::fabs(v);
                        if (j == 0 || a < min1) { min1 = a; i1 = j; } // first minimum, as argmin
                    }
                    double min2 = INF;
                    for (long long j = 0; j < d; j++) if (j != i1) min2 = std::min(min2, std::fabs(q[j]));
                    const double total_sign = (negs % 2 == 1) ? -1.0 : 1.0;
                    for (long long j = 0; j < d; j++) {
                        const double s = total_sign * (q[j] < 0 ? -1.0 : 1.0);
                        const double nr = alpha*s*(j == i1 ? min2 : min1);
                        r[e0 + j] = nr;
                        total[var[e0 + j]] = q[j] + nr;
                    }
                }
                done = true;
                for (long long c = 0; c < M && done; c++) {
                    int par = 0;
                    for (long long j = 0; j < deg[c]; j++) par ^= total[var[cstart[c] + j]] < 0 ? 1 : 0;
                    if (par) done = false;
                }
            }
            for (long long v = 0; v < N; v++) hd[b*N + v] = total[v] < 0;
            cv[b] = done;
        });
    }
    return py::make_tuple(hard, conv);
}

// ---------------------------------------------------------------- GF(2^m) and cyclic codes
// exp (2*size) / log (size) tables of rpps.math.gf.GF. Polynomials lowest degree first, as in rpps.math.cyclic.
struct GF {
    const int64_t *exp, *log;
    int64_t q; // size - 1

    int64_t mod(int64_t e) const { e %= q; return e < 0 ? e + q : e; }
    int64_t mul(int64_t a, int64_t b) const { return (a == 0 || b == 0) ? 0 : exp[mod(log[a] + log[b])]; }
    int64_t inv(int64_t a) const { return exp[mod(q - log[a])]; }
    int64_t pow_alpha(int64_t e) const { return exp[mod(e)]; }
    int64_t pow(int64_t a, int64_t e) const {
        if (e == 0) return 1;
        if (a == 0) return 0;
        return exp[mod(log[a]*e)];
    }
};

// S_i = r(alpha^((fcr + i)*prim)), i < nroots, position 0 the highest power of x
void syndromes(const GF &gf, const int64_t *w, long long N, int nroots, int fcr, int prim, int64_t *S) {
    for (int i = 0; i < nroots; i++) {
        int64_t acc = 0;
        const int64_t step = static_cast<int64_t>(fcr + i)*prim;
        for (long long k = 0; k < N; k++)
            if (w[k]) acc ^= gf.exp[gf.mod(gf.log[w[k]] + (N - 1 - k)*step)];
        S[i] = acc;
    }
}

void berlekamp_massey(const GF &gf, const int64_t *S, int nroots, const std::vector<int64_t> *gamma, std::vector<int64_t> &sigma, int &L) {
    sigma.assign(nroots + 1, 0);
    int e = 0;
    if (!gamma) {
        sigma[0] = 1;
    } else {
        e = static_cast<int>(gamma->size()) - 1;
        std::copy(gamma->begin(), gamma->end(), sigma.begin());
    }
    std::vector<int64_t> prev = sigma, T, shifted(nroots + 1);
    L = e;
    int m = 1;
    int64_t b = 1;
    for (int n = e; n < nroots; n++) {
        int64_t d = S[n];
        for (int i = 1; i <= std::min(L, n); i++) d ^= gf.mul(sigma[i], S[n - i]);
        if (d == 0) { m++; continue; }
        T = sigma;
        std::fill(shifted.begin(), shifted.end(), 0);
        const int64_t f = gf.mul(d, gf.inv(b));
        for (int i = m; i <= nroots; i++) shifted[i] = gf.mul(f, prev[i - m]);
        for (int i = 0; i <= nroots; i++) sigma[i] ^= shifted[i];
        if (2*L <= n + e) { L = n + 1 - L + e; prev = T; b = d; m = 1; }
        else m++;
    }
}

// Errors (and erasures, sorted unique positions) of one word of length N. false: uncorrectable
bool correct(const GF &gf, const int64_t *S, long long N, int nroots, int t, int fcr, int prim, bool binary,
             const std::vector<long long> &erasures, std::vector<long long> &pos, std::vector<int64_t> &val) {
    pos.clear(); val.clear();
    const int e = static_cast<int>(erasures.size());
    std::vector<int64_t> sigma;
    int L;
    if (e) {
        if (e > nroots) return false;
        std::vector<int64_t> g{1};
        for (long long p : erasures) {
            const int64_t X = gf.pow_alpha(gf.mod((N - 1 - p)*prim));
            std::vector<int64_t> nx(g.size() + 1, 0);
            for (std::size_t i = 0; i < g.size(); i++) nx[i] = g[i];
            for (std::size_t i = 0; i < g.size(); i++) nx[i + 1] ^= gf.mul(g[i], X);
            g.swap(nx);
        }
        berlekamp_massey(gf, S, nroots, &g, sigma, L);
        if (2*L - e > nroots) return false;
    } else {
        berlekamp_massey(gf, S, nroots, nullptr, sigma, L);
        if (L > t) return false;
    }
    for (long long k = 0; k < N; k++) { // Chien search
        const int64_t ek = (N - 1 - k)*prim;
        int64_t v = 0;
        for (int i = 0; i <= L; i++)
            if (sigma[i]) v ^= gf.exp[gf.mod(gf.log[sigma[i]] - ek*i)];
        if (v == 0) pos.push_back(k);
    }
    if (static_cast<int>(pos.size()) != L) return false;
    if (binary) { val.assign(L, 1); return true; }
    std::vector<int64_t> omega(nroots, 0); // Forney
    for (int i = 0; i < nroots; i++)
        for (int j = 0; j < std::min(i + 1, L + 1); j++) omega[i] ^= gf.mul(sigma[j], S[i - j]);
    for (long long p : pos) {
        const int64_t x = gf.pow_alpha((N - 1 - p)*prim), xinv = gf.inv(x);
        int64_t num = 0, den = 0;
        for (int k = 0; k < nroots; k++) num ^= gf.mul(omega[k], gf.pow(xinv, k));
        for (int i = 1; i <= L; i += 2) den ^= gf.mul(sigma[i], gf.pow(xinv, i - 1));
        if (den == 0) return false;
        val.push_back(gf.mul(gf.mul(num, gf.inv(den)), gf.pow(x, 1 - fcr)));
    }
    return true;
}

GF make_gf(carray<int64_t> &exp, carray<int64_t> &log) {
    const int64_t size = log.size();
    if (size < 2 || exp.size() < 2*size - 1) throw std::invalid_argument("GF tables: exp needs 2*size entries, log size");
    return GF{exp.data(), log.data(), size - 1};
}

void check_symbols(const int64_t *w, long long n, int64_t size) {
    for (long long k = 0; k < n; k++)
        if (w[k] < 0 || w[k] >= size) throw std::invalid_argument("symbol outside the field");
}

// Syndromes of many words: S_i = sum_k w_k alpha^((N-1-k)(fcr+i) prim), the exponent of every (i, k) taken mod q once
// for all words (a table) and exp doubled so log[w_k] + e_ik indexes it with no modulo: independent lookups a symbol,
// words in parallel. The same values as syndromes() above.
py::array_t<int64_t> gf_syndromes(carray<int64_t> words, carray<int64_t> exp, carray<int64_t> log, int nroots, int fcr, int prim,
                                  int threads) {
    GF gf = make_gf(exp, log);
    check_shape(words, {-1, -1}, "words");
    const long long W = words.shape(0), N = words.shape(1);
    check_symbols(words.data(), W*N, gf.q + 1);
    py::array_t<int64_t> S({W, static_cast<long long>(nroots)});
    const int64_t *w = words.data();
    int64_t *out = S.mutable_data();
    {
        py::gil_scoped_release nogil;
        const int64_t q = gf.q;
        std::vector<int64_t> exp2(2*q);
        for (int64_t e = 0; e < 2*q; e++) exp2[e] = gf.exp[e % q];
        std::vector<int64_t> e(static_cast<std::size_t>(nroots*N));
        for (int i = 0; i < nroots; i++) {
            const int64_t step = gf.mod(static_cast<int64_t>(fcr + i)*prim);
            int64_t v = 0; // (N-1-k) step mod q, from k = N-1 down
            for (long long k = N - 1; k >= 0; k--) { e[i*N + k] = v; v += step; if (v >= q) v -= q; }
        }
        const int64_t *lg = gf.log, *ex = exp2.data(), *et = e.data();
        parallel_for(W, W*N*nroots > (1 << 16) ? threads : 1, [&](long long r) {
            const int64_t *x = w + r*N;
            int64_t *o = out + r*nroots;
            for (int i = 0; i < nroots; i++) {
                const int64_t *ei = et + i*N;
                int64_t acc = 0;
                for (long long k = 0; k < N; k++)
                    if (x[k]) acc ^= ex[lg[x[k]] + ei[k]];
                o[i] = acc;
            }
        });
    }
    return S;
}

// One word from its syndromes: (positions, values), or None when it can't be corrected
py::object cyclic_correct(carray<int64_t> S, carray<int64_t> exp, carray<int64_t> log, long long N, int nroots, int t, int fcr,
                          int prim, bool binary, std::vector<long long> erasures) {
    GF gf = make_gf(exp, log);
    if (S.size() < nroots) throw std::invalid_argument("too few syndromes");
    std::sort(erasures.begin(), erasures.end());
    erasures.erase(std::unique(erasures.begin(), erasures.end()), erasures.end());
    if (binary && !erasures.empty()) throw std::invalid_argument("Erasures need a non-binary code (Reed-Solomon)");
    std::vector<long long> pos;
    std::vector<int64_t> val;
    bool ok;
    {
        py::gil_scoped_release nogil;
        ok = correct(gf, S.data(), N, nroots, t, fcr, prim, binary, erasures, pos, val);
    }
    if (!ok) return py::none();
    auto p = empty<int64_t>(pos.size()), v = empty<int64_t>(val.size());
    std::copy(pos.begin(), pos.end(), p.mutable_data());
    std::copy(val.begin(), val.end(), v.mutable_data());
    return py::make_tuple(p, v);
}

// Every word (W, N) decoded in place: errors[w] = 0 (syndromes zero), -1 (uncorrectable), else the corrected count
// (non-zero error values for RS, bit flips for binary). erased (W, N): positions known unreliable (none: empty).
// verify: re-check the corrected word's syndromes (Reed-Solomon does).
py::tuple cyclic_decode(carray<int64_t> words, carray<uint8_t> erased, carray<int64_t> exp, carray<int64_t> log, int nroots, int t,
                        int fcr, int prim, bool binary, bool verify, int threads) {
    GF gf = make_gf(exp, log);
    check_shape(words, {-1, -1}, "words");
    const long long W = words.shape(0), N = words.shape(1);
    const bool have_erased = erased.size() > 0;
    if (have_erased) check_shape(erased, {W, N}, "erased");
    if (binary && have_erased) throw std::invalid_argument("Erasures need a non-binary code (Reed-Solomon)");
    check_symbols(words.data(), W*N, gf.q + 1);
    py::array_t<int64_t> out({W, N});
    auto errors = empty<int64_t>(static_cast<std::size_t>(W));
    int64_t *o = out.mutable_data(), *er = errors.mutable_data();
    std::copy(words.data(), words.data() + W*N, o);
    const uint8_t *ers = erased.data();
    {
        py::gil_scoped_release nogil;
        parallel_for(W, threads, [&](long long w) {
            int64_t *word = o + w*N;
            std::vector<int64_t> S(nroots), S2(nroots), val;
            std::vector<long long> pos, eras;
            syndromes(gf, word, N, nroots, fcr, prim, S.data());
            if (std::all_of(S.begin(), S.end(), [](int64_t s) { return s == 0; })) { er[w] = 0; return; }
            if (have_erased)
                for (long long k = 0; k < N; k++) if (ers[w*N + k]) eras.push_back(k);
            if (!correct(gf, S.data(), N, nroots, t, fcr, prim, binary, eras, pos, val)) { er[w] = -1; return; }
            std::vector<int64_t> fixed(word, word + N);
            for (std::size_t i = 0; i < pos.size(); i++) fixed[pos[i]] ^= val[i];
            if (verify) {
                syndromes(gf, fixed.data(), N, nroots, fcr, prim, S2.data());
                if (std::any_of(S2.begin(), S2.end(), [](int64_t s) { return s != 0; })) { er[w] = -1; return; }
            }
            std::copy(fixed.begin(), fixed.end(), word);
            er[w] = binary ? static_cast<int64_t>(pos.size())
                           : static_cast<int64_t>(std::count_if(val.begin(), val.end(), [](int64_t v) { return v != 0; }));
        });
    }
    return py::make_tuple(out, errors);
}

} // namespace

void register_fec(py::module_ &m) {
    m.def("ldpc_minsum", &ldpc_minsum, "ldpc_minsum(llr (B, N), var, cstart, deg, rows (check order: the layers), iters, alpha, threads) -> (hard (B, N), converged (B))",
          py::arg("llr"), py::arg("var"), py::arg("cstart"), py::arg("deg"), py::arg("rows"), py::arg("iters"), py::arg("alpha"),
          py::arg("threads") = 0);
    m.def("gf_syndromes", &gf_syndromes, "gf_syndromes(words (W, N), exp, log, nroots, fcr, prim, threads) -> S (W, nroots)",
          py::arg("words"), py::arg("exp"), py::arg("log"), py::arg("nroots"), py::arg("fcr"), py::arg("prim"), py::arg("threads") = 0);
    m.def("cyclic_correct", &cyclic_correct, "cyclic_correct(S, exp, log, N, nroots, t, fcr, prim, binary, erasures) -> (positions, values) or None",
          py::arg("S"), py::arg("exp"), py::arg("log"), py::arg("N"), py::arg("nroots"), py::arg("t"), py::arg("fcr"), py::arg("prim"),
          py::arg("binary"), py::arg("erasures"));
    m.def("cyclic_decode", &cyclic_decode, "cyclic_decode(words (W, N), erased (W, N) or empty, exp, log, nroots, t, fcr, prim, binary, verify, threads) -> (words, errors)",
          py::arg("words"), py::arg("erased"), py::arg("exp"), py::arg("log"), py::arg("nroots"), py::arg("t"), py::arg("fcr"), py::arg("prim"),
          py::arg("binary"), py::arg("verify"), py::arg("threads") = 0);
}

} // namespace rpps
