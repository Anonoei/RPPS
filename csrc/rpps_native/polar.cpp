// Polar successive-cancellation list decoding (rpps/code/types/polar.py scl, the reference): every frame of a batch on
// several threads. The same tree as the reference, in natural order (no bit reversal): a node of m LLRs gives its left
// child box(a, c) (exact box-plus, clipped as the reference clips) and its right child c + (1 - 2v) a, and returns
// (v ^ w, w). A path's metric is the penalty sum log(1 + e^(-(1 - 2u) L)); at an information bit the 2P candidates
// (all paths with 0, then all with 1) are ranked by metric, stable, and the best `size` kept, as numpy's stable argsort
// keeps them. A path is one buffer per tree depth (LLRs, and the bits of the left child a node keeps), copied when the
// survivors are chosen: N is at most a few thousand and the list a few paths, so copying is cheaper than sharing.
#include "common.hpp"
#include "parallel.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <vector>

namespace rpps {

namespace {

inline double clip(double x, double lo, double hi) { return x < lo ? lo : (x > hi ? hi : x); }

// The reference's exact box-plus: 2 atanh(tanh(a/2) tanh(b/2)), arguments clipped to +-30, the product to +-(1 - 1e-15)
inline double box(double a, double b) {
    const double t = std::tanh(clip(a, -30.0, 30.0)/2)*std::tanh(clip(b, -30.0, 30.0)/2);
    return 2*std::atanh(clip(t, -1 + 1e-15, 1 - 1e-15));
}

// numpy's logaddexp(x1, x2)
inline double logaddexp(double x1, double x2) {
    if (x1 == x2) return x1 + 0.69314718055994530942;
    const double d = x1 - x2;
    return d > 0 ? x1 + std::log1p(std::exp(-d)) : x2 + std::log1p(std::exp(d));
}

struct Path {
    std::vector<std::vector<double>> L; // L[d]: the LLRs of the node at depth d (N >> d of them)
    std::vector<std::vector<uint8_t>> X; // X[d]: the bits a node at depth d returns
    std::vector<std::vector<uint8_t>> V; // V[d]: a node's left child's bits, kept while its right child runs
    std::vector<uint8_t> u;
    double pm = 0;
};

struct Decoder {
    int n, N, size;
    const uint8_t *frozen;
    std::vector<Path> paths, next;
    std::vector<double> cand;
    std::vector<int> idx;

    Decoder(int n_, const uint8_t *frozen_, int size_) : n(n_), N(1 << n_), size(size_), frozen(frozen_) {}

    Path blank() const {
        Path p;
        p.L.resize(n + 1); p.X.resize(n + 1); p.V.resize(n + 1);
        for (int d = 0; d <= n; d++) {
            p.L[d].assign(static_cast<size_t>(N >> d), 0.0);
            p.X[d].assign(static_cast<size_t>(N >> d), 0);
            p.V[d].assign(static_cast<size_t>(std::max(1, (N >> d)/2)), 0);
        }
        p.u.assign(static_cast<size_t>(N), 0);
        return p;
    }

    // Decide leaf i0 (every path's L[n][0]); paths may be replaced by survivors
    void leaf(int i0) {
        const int P = static_cast<int>(paths.size());
        if (frozen[i0]) {
            for (auto &p : paths) {
                p.pm += logaddexp(0.0, -p.L[n][0]);
                p.u[i0] = 0; p.X[n][0] = 0;
            }
            return;
        }
        cand.resize(static_cast<size_t>(2*P));
        for (int q = 0; q < P; q++) {
            const double Li = paths[q].L[n][0];
            cand[q] = paths[q].pm + logaddexp(0.0, -Li);
            cand[P + q] = paths[q].pm + logaddexp(0.0, Li);
        }
        idx.resize(static_cast<size_t>(2*P));
        std::iota(idx.begin(), idx.end(), 0);
        std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) { return cand[a] < cand[b]; });
        const int keep = std::min(2*P, size);
        next.clear();
        for (int j = 0; j < keep; j++) {
            const int c = idx[j], q = c % P;
            const uint8_t bit = static_cast<uint8_t>(c/P);
            next.push_back(paths[q]); // (a copy: two survivors may come from one path)
            Path &p = next.back();
            p.pm = cand[c];
            p.u[i0] = bit; p.X[n][0] = bit;
        }
        paths.swap(next);
    }

    // The node at depth d whose first bit is i0: its LLRs are in every path's L[d]; it leaves its bits in X[d]
    void node(int d, int i0) {
        if (d == n) {
            leaf(i0);
            return;
        }
        const int m = N >> d, h = m/2;
        for (auto &p : paths) {
            const double *a = p.L[d].data(), *c = a + h;
            double *o = p.L[d + 1].data();
            for (int k = 0; k < h; k++) o[k] = box(a[k], c[k]);
        }
        node(d + 1, i0);
        for (auto &p : paths) { // The left child's bits, then the right child's LLRs
            std::memcpy(p.V[d].data(), p.X[d + 1].data(), static_cast<size_t>(h));
            const double *a = p.L[d].data(), *c = a + h;
            const uint8_t *v = p.V[d].data();
            double *o = p.L[d + 1].data();
            for (int k = 0; k < h; k++) o[k] = c[k] + (1 - 2.0*v[k])*a[k];
        }
        node(d + 1, i0 + h);
        for (auto &p : paths) {
            const uint8_t *v = p.V[d].data(), *w = p.X[d + 1].data();
            uint8_t *x = p.X[d].data();
            for (int k = 0; k < h; k++) { x[k] = v[k] ^ w[k]; x[h + k] = w[k]; }
        }
    }

    void run(const double *llr) {
        paths.assign(1, blank());
        std::memcpy(paths[0].L[0].data(), llr, sizeof(double)*static_cast<size_t>(N));
        node(0, 0);
    }
};

// scl: (llr (W, N), frozen (N), size) -> (u (W, P, N), metric (W, P)), each frame's paths best first
py::tuple polar_scl(carray<double> llr, carray<bool> frozen_a, int size, int threads) {
    check_shape(llr, {-1, -1}, "llr");
    const long long W = llr.shape(0), N = llr.shape(1);
    int n = 0;
    while ((1LL << n) < N) n++;
    if ((1LL << n) != N || N < 1) throw std::invalid_argument("polar: N is not a power of 2");
    check_shape(frozen_a, {N}, "frozen");
    if (size < 1) throw std::invalid_argument("polar: list size is at least 1");
    std::vector<uint8_t> frozen(static_cast<size_t>(N));
    for (long long i = 0; i < N; i++) frozen[i] = frozen_a.data()[i] ? 1 : 0;
    long long info = 0;
    for (auto f : frozen) info += f ? 0 : 1;
    long long P = 1;
    for (long long i = 0; i < info && P < size; i++) P = std::min<long long>(2*P, size); // Paths at the end
    py::array_t<uint8_t> u({W, P, N});
    py::array_t<double> pm({W, P});
    const double *in = llr.data();
    uint8_t *uo = u.mutable_data();
    double *po = pm.mutable_data();
    {
        py::gil_scoped_release nogil;
        parallel_for(W, threads, [&](long long w) {
            Decoder dec(n, frozen.data(), size);
            dec.run(in + w*N);
            const long long np_ = static_cast<long long>(dec.paths.size());
            std::vector<int> order(static_cast<size_t>(np_));
            std::iota(order.begin(), order.end(), 0);
            std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return dec.paths[a].pm < dec.paths[b].pm; });
            for (long long j = 0; j < P; j++) {
                const Path &p = dec.paths[order[std::min<long long>(j, np_ - 1)]];
                std::memcpy(uo + (w*P + j)*N, p.u.data(), static_cast<size_t>(N));
                po[w*P + j] = p.pm;
            }
        });
    }
    return py::make_tuple(u, pm);
}

} // namespace

void register_polar(py::module_ &m) {
    m.def("polar_scl", &polar_scl, "polar_scl(llr (W, N), frozen (N) bool, size, threads) -> (u (W, P, N) uint8, metric (W, P)), best first",
          py::arg("llr"), py::arg("frozen"), py::arg("size"), py::arg("threads") = 0);
}

} // namespace rpps
