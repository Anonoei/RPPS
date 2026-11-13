// Trellis decoders: Viterbi (rpps/coding/types/viterbi.py, fsm.py) and BCJR forward-backward (trellis.py: turbo,
// bcjr_conv, syndrome trellises). Generic over the code: the Python classes hand over their tables.
#include "common.hpp"
#include "parallel.hpp"

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace rpps {

namespace {

constexpr double INF = std::numeric_limits<double>::infinity();

// ---------------------------------------------------------------- Viterbi
// A step t has m inputs x[t]. Branch j into state s comes from pred[s][j] and sends pattern pat[s][j]; the
// pattern's cost is, by `metric`: 0 the squared distance sum_i (x_i - tab[c][i])^2 (amplitudes), 1 the dot product
// sum_i x_i tab[c][i] (LLRs against the pattern's 1 bits). The first of equal survivors wins, as numpy's argmin.
struct Trellis {
    int S, P, C, m, metric;
    const int32_t *pred, *pat;
    const double *tab;

    void costs(const double *x, double *cost) const {
        for (int c = 0; c < C; c++) {
            const double *t = tab + static_cast<std::size_t>(c)*m;
            double acc = 0.0;
            if (metric == 0) {
                for (int i = 0; i < m; i++) { const double d = x[i] - t[i]; acc += d*d; }
            } else {
                for (int i = 0; i < m; i++) acc += x[i]*t[i];
            }
            cost[c] = acc;
        }
    }

    // T steps from path metrics pm (S, updated): decisions dec (T, S)
    void forward(const double *x, long long T, double *pm, uint8_t *dec) const {
        std::vector<double> cost(C), nm(S);
        for (long long t = 0; t < T; t++) {
            costs(x + t*m, cost.data());
            double mn = INF;
            for (int s = 0; s < S; s++) {
                double best = INF; int bj = 0;
                for (int j = 0; j < P; j++) {
                    const double c = pm[pred[s*P + j]] + cost[pat[s*P + j]];
                    if (c < best) { best = c; bj = j; }
                }
                nm[s] = best;
                dec[t*S + s] = static_cast<uint8_t>(bj);
                if (best < mn) mn = best;
            }
            for (int s = 0; s < S; s++) pm[s] = nm[s] - mn;
        }
    }

    // Path back from state `end` after the last step: states[t] (the state after step t) and the branch taken
    void traceback(const uint8_t *dec, long long T, int end, int32_t *states, uint8_t *branch) const {
        int s = end;
        for (long long t = T - 1; t >= 0; t--) {
            const int j = dec[t*S + s];
            states[t] = s;
            branch[t] = static_cast<uint8_t>(j);
            s = pred[s*P + j];
        }
    }
};

Trellis make_trellis(carray<double> &tab, int metric, carray<int32_t> &pred, carray<int32_t> &pat) {
    check_shape(tab, {-1, -1}, "tab");
    check_shape(pred, {-1, -1}, "pred");
    check_shape(pat, {pred.shape(0), pred.shape(1)}, "pat");
    Trellis tr{static_cast<int>(pred.shape(0)), static_cast<int>(pred.shape(1)), static_cast<int>(tab.shape(0)),
               static_cast<int>(tab.shape(1)), metric, pred.data(), pat.data(), tab.data()};
    if (tr.P > 255) throw std::invalid_argument("more than 255 branches into a state");
    for (py::ssize_t i = 0; i < pred.size(); i++) {
        if (tr.pred[i] < 0 || tr.pred[i] >= tr.S || tr.pat[i] < 0 || tr.pat[i] >= tr.C)
            throw std::invalid_argument("pred/pat entries out of range");
    }
    return tr;
}

py::tuple viterbi_forward(carray<double> x, carray<double> tab, int metric, carray<int32_t> pred, carray<int32_t> pat, carray<double> pm_a) {
    Trellis tr = make_trellis(tab, metric, pred, pat);
    check_shape(x, {-1, tr.m}, "x");
    check_shape(pm_a, {tr.S}, "pm");
    const long long T = x.shape(0);
    py::array_t<uint8_t> dec({static_cast<py::ssize_t>(T), static_cast<py::ssize_t>(tr.S)});
    auto pm = empty<double>(static_cast<std::size_t>(tr.S));
    std::copy(pm_a.data(), pm_a.data() + tr.S, pm.mutable_data());
    {
        py::gil_scoped_release nogil;
        tr.forward(x.data(), T, pm.mutable_data(), dec.mutable_data());
    }
    return py::make_tuple(dec, pm);
}

py::tuple viterbi_traceback(carray<uint8_t> dec, carray<int32_t> pred, int end) {
    check_shape(dec, {-1, pred.shape(0)}, "dec");
    const long long T = dec.shape(0);
    const int S = static_cast<int>(pred.shape(0)), P = static_cast<int>(pred.shape(1));
    if (end < 0 || end >= S) throw std::invalid_argument("end state out of range");
    Trellis tr{S, P, 0, 0, 0, pred.data(), nullptr, nullptr};
    auto states = empty<int32_t>(static_cast<std::size_t>(T));
    auto branch = empty<uint8_t>(static_cast<std::size_t>(T));
    {
        py::gil_scoped_release nogil;
        for (long long i = 0; i < T*S; i++)
            if (dec.data()[i] >= P) throw std::invalid_argument("decision out of range");
        tr.traceback(dec.data(), T, end, states.mutable_data(), branch.mutable_data());
    }
    return py::make_tuple(states, branch);
}

// W independent sequences (W, T, m), all from path metrics pm0 (S); end (W): the final state, -1 = the best one
py::tuple viterbi_batch(carray<double> x, carray<double> tab, int metric, carray<int32_t> pred, carray<int32_t> pat,
                        carray<double> pm0, carray<int64_t> end, int threads) {
    Trellis tr = make_trellis(tab, metric, pred, pat);
    check_shape(x, {-1, -1, tr.m}, "x");
    check_shape(pm0, {tr.S}, "pm0");
    const long long W = x.shape(0), T = x.shape(1);
    check_shape(end, {W}, "end");
    py::array_t<int32_t> states({W, T});
    py::array_t<uint8_t> branch({W, T});
    py::array_t<double> pm_out({W, static_cast<long long>(tr.S)});
    const double *xs = x.data(), *p0 = pm0.data();
    const int64_t *ends = end.data();
    int32_t *st = states.mutable_data();
    uint8_t *br = branch.mutable_data();
    double *pmo = pm_out.mutable_data();
    for (long long w = 0; w < W; w++)
        if (ends[w] >= tr.S) throw std::invalid_argument("end state out of range");
    {
        py::gil_scoped_release nogil;
        parallel_for(W, threads, [&](long long w) {
            std::vector<uint8_t> dec(static_cast<std::size_t>(T)*tr.S);
            double *pm = pmo + w*tr.S;
            std::copy(p0, p0 + tr.S, pm);
            tr.forward(xs + w*T*tr.m, T, pm, dec.data());
            int e = static_cast<int>(ends[w]);
            if (e < 0) {
                double b = INF; e = 0;
                for (int s = 0; s < tr.S; s++) if (pm[s] < b) { b = pm[s]; e = s; }
            }
            if (T > 0) tr.traceback(dec.data(), T, e, st + w*T, br + w*T);
        });
    }
    return py::make_tuple(states, branch, pm_out);
}

// ---------------------------------------------------------------- BCJR
// numpy's logaddexp and NaN-propagating max, so the results match the Python reference (nan_to_num included)
inline double logaddexp(double x, double y) {
    if (x == y) return x + 0.6931471805599453;
    const double d = x - y;
    if (d > 0) return x + std::log1p(std::exp(-d));
    if (d <= 0) return y + std::log1p(std::exp(d));
    return d; // NaN
}

inline double nanmax(const double *v, int n) {
    double m = v[0];
    for (int i = 0; i < n; i++) {
        if (std::isnan(v[i])) return v[i];
        if (v[i] > m) m = v[i];
    }
    return m;
}

inline double nan_to_num(double v) {
    if (std::isnan(v)) return 0.0;
    if (std::isinf(v)) return v > 0 ? 1e4 : -1e4;
    return v;
}

// llr (W, T, L); sign (P, S, 2, L): +1 for a 0 bit; next/pred_state/pred_bit (P, S, 2). Section t uses table t % P.
py::array_t<double> bcjr(carray<double> llr, carray<double> sign, carray<int64_t> next, carray<int64_t> pred_state,
                         carray<int64_t> pred_bit, long long start, long long end, bool maxlog, int threads) {
    check_shape(llr, {-1, -1, -1}, "llr");
    const long long W = llr.shape(0), T = llr.shape(1), L = llr.shape(2);
    check_shape(sign, {-1, -1, 2, L}, "sign");
    const long long P = sign.shape(0), S = sign.shape(1);
    check_shape(next, {P, S, 2}, "next");
    check_shape(pred_state, {P, S, 2}, "pred_state");
    check_shape(pred_bit, {P, S, 2}, "pred_bit");
    for (py::ssize_t i = 0; i < next.size(); i++) {
        if (next.data()[i] < 0 || next.data()[i] >= S || pred_state.data()[i] < 0 || pred_state.data()[i] >= S ||
            pred_bit.data()[i] < 0 || pred_bit.data()[i] > 1)
            throw std::invalid_argument("trellis tables out of range");
    }
    if (start >= S || end >= S) throw std::invalid_argument("start/end state out of range");
    py::array_t<double> out({W, T});
    const double *lp = llr.data(), *sg = sign.data();
    const int64_t *nx = next.data(), *ps = pred_state.data(), *pb = pred_bit.data();
    double *op = out.mutable_data();
    {
        py::gil_scoped_release nogil;
        parallel_for(W, threads, [&](long long w) {
            const double *x = lp + w*T*L;
            std::vector<double> alpha(static_cast<std::size_t>((T + 1)*S)), beta(S), nb(S), g(2*S), m0(S), m1(S);
            auto gamma = [&](long long t) {
                const double *sgp = sg + (t % P)*S*2*L, *xt = x + t*L;
                for (long long s = 0; s < S; s++)
                    for (int i = 0; i < 2; i++) {
                        double acc = 0.0;
                        for (long long l = 0; l < L; l++) acc += sgp[(s*2 + i)*L + l]*xt[l];
                        g[s*2 + i] = 0.5*acc;
                    }
            };
            for (long long s = 0; s < S; s++) alpha[s] = (start < 0 || s == start) ? 0.0 : -INF;
            for (long long t = 0; t < T; t++) {
                gamma(t);
                const long long p = t % P;
                const double *a = alpha.data() + t*S;
                double *an = alpha.data() + (t + 1)*S;
                for (long long s = 0; s < S; s++) {
                    const long long s0 = ps[(p*S + s)*2], s1 = ps[(p*S + s)*2 + 1];
                    const long long b0 = pb[(p*S + s)*2], b1 = pb[(p*S + s)*2 + 1];
                    const double u = a[s0] + g[s0*2 + b0], v = a[s1] + g[s1*2 + b1];
                    an[s] = maxlog ? std::max(u, v) : logaddexp(u, v);
                    if (maxlog && (std::isnan(u) || std::isnan(v))) an[s] = std::isnan(u) ? u : v;
                }
                const double mx = nanmax(an, static_cast<int>(S));
                for (long long s = 0; s < S; s++) an[s] -= mx;
            }
            for (long long s = 0; s < S; s++) beta[s] = (end < 0 || s == end) ? 0.0 : -INF;
            for (long long t = T - 1; t >= 0; t--) {
                gamma(t);
                const long long p = t % P;
                const double *a = alpha.data() + t*S;
                const int64_t *nxt = nx + p*S*2;
                for (long long s = 0; s < S; s++) {
                    m0[s] = a[s] + g[s*2] + beta[nxt[s*2]];
                    m1[s] = a[s] + g[s*2 + 1] + beta[nxt[s*2 + 1]];
                }
                double r0, r1;
                if (maxlog) {
                    r0 = nanmax(m0.data(), static_cast<int>(S)); r1 = nanmax(m1.data(), static_cast<int>(S));
                } else {
                    r0 = m0[0]; r1 = m1[0];
                    for (long long s = 1; s < S; s++) { r0 = logaddexp(r0, m0[s]); r1 = logaddexp(r1, m1[s]); }
                }
                op[w*T + t] = nan_to_num(r0 - r1);
                for (long long s = 0; s < S; s++) {
                    const double u = g[s*2] + beta[nxt[s*2]], v = g[s*2 + 1] + beta[nxt[s*2 + 1]];
                    nb[s] = maxlog ? std::max(u, v) : logaddexp(u, v);
                    if (maxlog && (std::isnan(u) || std::isnan(v))) nb[s] = std::isnan(u) ? u : v;
                }
                const double mx = nanmax(nb.data(), static_cast<int>(S));
                for (long long s = 0; s < S; s++) beta[s] = nb[s] - mx;
            }
        });
    }
    return out;
}

} // namespace

void register_trellis(py::module_ &m) {
    m.def("viterbi_forward", &viterbi_forward, "viterbi_forward(x (T, m), tab (C, m), metric, pred (S, P), pat (S, P), pm (S)) -> (dec (T, S), pm)",
          py::arg("x"), py::arg("tab"), py::arg("metric"), py::arg("pred"), py::arg("pat"), py::arg("pm"));
    m.def("viterbi_traceback", &viterbi_traceback, "viterbi_traceback(dec (T, S), pred (S, P), end) -> (states (T), branch (T))",
          py::arg("dec"), py::arg("pred"), py::arg("end"));
    m.def("viterbi_batch", &viterbi_batch, "viterbi_batch(x (W, T, m), tab, metric, pred, pat, pm0 (S), end (W), threads) -> (states (W, T), branch (W, T), pm (W, S))",
          py::arg("x"), py::arg("tab"), py::arg("metric"), py::arg("pred"), py::arg("pat"), py::arg("pm0"), py::arg("end"), py::arg("threads") = 0);
    m.def("bcjr", &bcjr, "bcjr(llr (W, T, L), sign (P, S, 2, L), next, pred_state, pred_bit (P, S, 2), start, end, maxlog, threads) -> app (W, T)",
          py::arg("llr"), py::arg("sign"), py::arg("next"), py::arg("pred_state"), py::arg("pred_bit"), py::arg("start"), py::arg("end"),
          py::arg("maxlog"), py::arg("threads") = 0);
}

} // namespace rpps
