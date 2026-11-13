// Frame sync, the header-word metric (rpps/sync/frame.py): for every candidate preamble position, the best correlation of
// the symbols after it with a set of known header words. One call over all candidates, on several threads, with no matrix
// product library underneath (a BLAS threads inside a call and spins between calls, which slows everything around it).
#include "common.hpp"
#include "parallel.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace rpps {

namespace {

#if defined(__GNUC__) && defined(__x86_64__) && !defined(__clang__)
#define RPPS_CLONES __attribute__((target_clones("avx2,fma", "default")))
#else
#define RPPS_CLONES
#endif

// max over the words m of dot(t, w[m]); t and the words are 2n floats (re, im pairs, the words already conjugated and signed
// so a plain dot is Re(sum t_k * word_k))
RPPS_CLONES
static float best_word(const float *t, const float *w, long long M, long long n2) {
    float best = -3.4e38f;
    for (long long m = 0; m < M; m++) {
        const float *wm = w + m*n2;
        float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        long long k = 0;
        for (; k + 8 <= n2; k += 8)
            for (int i = 0; i < 8; i++) acc[i] += t[k + i]*wm[k + i];
        float s = ((acc[0] + acc[4]) + (acc[1] + acc[5])) + ((acc[2] + acc[6]) + (acc[3] + acc[7]));
        for (; k < n2; k++) s += t[k]*wm[k];
        best = s > best ? s : best;
    }
    return best;
}

// The whole word stage: for every position j whose preamble metric m[j] passes `cand` (and whose continuation is all in y), the
// metric over the preamble and its best continuation, (|c| + best word) / sqrt(E (L + n)), E the energy of the L + n symbols;
// 0 elsewhere. y starts where c does.
py::array_t<double> frame_metric(carray<cf64> y_a, carray<cf64> c_a, carray<double> m_a, long long L, carray<float> w_a, double cand, int threads) {
    check_shape(w_a, {-1, -1}, "words");
    const long long nw = m_a.size(), M = w_a.shape(0), n2 = w_a.shape(1), n = n2/2, S = y_a.size();
    if (c_a.size() != nw) throw std::invalid_argument("c and m: one per position");
    const cf64 *y = y_a.data(), *c = c_a.data();
    const double *m = m_a.data();
    const float *w = w_a.data();
    py::array_t<double> out(static_cast<py::ssize_t>(nw));
    double *o = out.mutable_data();
    constexpr long long CHUNK = 4096;
    {
        py::gil_scoped_release nogil;
        parallel_for((nw + CHUNK - 1)/CHUNK, threads, [&](long long blk) {
            std::vector<float> t(static_cast<size_t>(n2));
            const long long a = blk*CHUNK, end = std::min(nw, a + CHUNK);
            for (long long j = a; j < end; j++) {
                if (!(m[j] > cand) || j + L + n > S) { o[j] = 0.0; continue; }
                const double cm = std::sqrt(c[j].real()*c[j].real() + c[j].imag()*c[j].imag());
                const double inv = 1.0/std::max(cm, 1e-30);
                const double ur = c[j].real()*inv, ui = -c[j].imag()*inv; // conj(c/|c|): the phase taken off
                const cf64 *s = y + j + L;
                for (long long k = 0; k < n; k++) {
                    t[2*k] = static_cast<float>(s[k].real()*ur - s[k].imag()*ui);
                    t[2*k + 1] = static_cast<float>(s[k].real()*ui + s[k].imag()*ur);
                }
                const double best = best_word(t.data(), w, M, n2);
                double e = 0.0;
                for (long long k = 0; k < L + n; k++) e += y[j + k].real()*y[j + k].real() + y[j + k].imag()*y[j + k].imag();
                o[j] = (cm + best)/std::sqrt(std::max(e, 1e-30)*static_cast<double>(L + n));
            }
        });
    }
    return out;
}

} // namespace

void register_framesync(py::module_ &m) {
    m.def("frame_metric", &frame_metric,
          "frame_metric(y complex128, c complex128, m, L, words (M, 2n) float32, candidate, threads) -> the metric over the preamble and its best word at every candidate, 0 elsewhere",
          py::arg("y"), py::arg("c"), py::arg("m"), py::arg("L"), py::arg("words"), py::arg("candidate"), py::arg("threads") = 0);
}

} // namespace rpps
