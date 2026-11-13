// Fractional polyphase resampling (rpps/signal/sample/fractional.py): a table of interpolation kernels, one per fraction of a sample,
// read at positions pos0 + k*step; and the frequency shift that goes in front of it.
//
// polyphase_read: out[k] = sum_j table[ph(k)][j] * buf[i0(k) - half + 1 + j], i0 = floor(pos0 + k*step), ph = the nearest row to
// the fraction. Outputs are independent, so they run on several threads. rotate_into: out[i] = x[i] exp(j (phase0 + w i)), the
// chunks of a block on several threads, each starting from an exact phase.
#include "common.hpp"
#include "parallel.hpp"

#include <algorithm>
#include <cmath>
#include <type_traits>
#include <vector>

namespace rpps {

namespace {

// Dispatch on the CPU's features, not its model: `arch=x86-64-v3` (AVX2 + FMA + BMI) is chosen on any CPU that has them. `arch=haswell` and the other
// named CPUs are matched by model, and on an AMD CPU the resolver falls through to the SSE2 clone (measured: 2.5x slower on a Zen 5)
#if defined(__GNUC__) && defined(__x86_64__) && !defined(__clang__) && __GNUC__ >= 12
#define RPPS_CLONES __attribute__((target_clones("arch=x86-64-v3", "default")))
#else
#define RPPS_CLONES
#endif

// 64-bit loop indices throughout: with -fno-strict-overflow (CPython's own flags) 32-bit ones are not widened and the loops run 3x slower
//
// Both reads are written with GCC / Clang vector types: the sums stay in vector registers and the lanes are folded as a vector does (high half onto low, then
// high pair onto low), where a loop over an array of lanes is folded by the compiler one scalar add at a time (35 instructions against 4) because floating point
// addition may not be reordered without fast-math. The vector form is the fold that was asked for, so it needs no fast-math and gives the same result everywhere.
// Elsewhere (MSVC) the plain loops below the #else are used: the same sums, slower.
#if defined(__GNUC__)
typedef float v8f __attribute__((vector_size(32), aligned(4))); // unaligned loads
typedef float v4f __attribute__((vector_size(16), aligned(4)));

static inline const v8f &ld(const float *p) { return *reinterpret_cast<const v8f *>(p); }
static inline float fold8(v8f a, float *odd) { // lanes 0, 2, 4, 6 sum to the return value, 1, 3, 5, 7 to *odd
    v4f w = {a[0], a[1], a[2], a[3]}; const v4f u = {a[4], a[5], a[6], a[7]};
    w += u;
    *odd = w[1] + w[3];
    return w[0] + w[2];
}
#endif

// Complex: s is interleaved (re, im) pairs and t2 the taps with each repeated (t, t, ...), so the products are one contiguous multiply per float and the lanes
// alternate re, im: no shuffling in the loop, and folding the lanes gives (re, im) directly
RPPS_CLONES
static void read_complex(const float *buf, const float *tab2, long long taps, int nph, double pos0, double step, long long k0, long long k1, float *out) {
    const long long half = taps/2, F = 2*taps;
    for (long long k = k0; k < k1; k++) {
        const double p = pos0 + static_cast<double>(k)*step;
        const long long i0 = static_cast<long long>(p);
        const int ph = static_cast<int>((p - static_cast<double>(i0))*nph + 0.5);
        const float *t = tab2 + static_cast<long long>(ph)*F;
        const float *s = buf + 2*(i0 - half + 1);
        long long j = 0;
        float sr, si;
#if defined(__GNUC__)
        v8f a0 = {}, a1 = {};
        for (; j + 16 <= F; j += 16) { a0 += ld(s + j)*ld(t + j); a1 += ld(s + j + 8)*ld(t + j + 8); }
        for (; j + 8 <= F; j += 8) a0 += ld(s + j)*ld(t + j);
        sr = fold8(a0 + a1, &si);
#else
        float acc[16] = {0};
        for (; j + 16 <= F; j += 16)
            for (long long l = 0; l < 16; l++) acc[l] += s[j + l]*t[j + l];
        sr = ((acc[0] + acc[8]) + (acc[2] + acc[10])) + ((acc[4] + acc[12]) + (acc[6] + acc[14]));
        si = ((acc[1] + acc[9]) + (acc[3] + acc[11])) + ((acc[5] + acc[13]) + (acc[7] + acc[15]));
#endif
        for (; j < F; j += 2) { sr += s[j]*t[j]; si += s[j + 1]*t[j + 1]; }
        out[2*k] = sr; out[2*k + 1] = si;
    }
}

RPPS_CLONES
static void read_real(const float *buf, const float *tab, long long taps, int nph, double pos0, double step, long long k0, long long k1, float *out) {
    const long long half = taps/2;
    for (long long k = k0; k < k1; k++) {
        const double p = pos0 + static_cast<double>(k)*step;
        const long long i0 = static_cast<long long>(p);
        const int ph = static_cast<int>((p - static_cast<double>(i0))*nph + 0.5);
        const float *t = tab + static_cast<long long>(ph)*taps;
        const float *s = buf + (i0 - half + 1);
        long long j = 0;
        float sum;
#if defined(__GNUC__)
        v8f a0 = {}, a1 = {};
        for (; j + 16 <= taps; j += 16) { a0 += ld(s + j)*ld(t + j); a1 += ld(s + j + 8)*ld(t + j + 8); }
        for (; j + 8 <= taps; j += 8) a0 += ld(s + j)*ld(t + j);
        float odd;
        sum = fold8(a0 + a1, &odd) + odd;
#else
        float a[8] = {0};
        for (; j + 8 <= taps; j += 8)
            for (long long l = 0; l < 8; l++) a[l] += s[j + l]*t[j + l];
        sum = ((a[0] + a[4]) + (a[1] + a[5])) + ((a[2] + a[6]) + (a[3] + a[7]));
#endif
        for (; j < taps; j++) sum += s[j]*t[j];
        out[k] = sum;
    }
}

template <typename E>
py::array_t<E> polyphase_read(py::array_t<E, py::array::c_style> buf_a, carray<float> table_a, double pos0, double step, long long nout, int threads) {
    if (table_a.ndim() != 2 || table_a.shape(0) < 2 || table_a.shape(1) < 2 || table_a.shape(1) % 2)
        throw std::invalid_argument("polyphase_read: table must be (phases + 1, taps), taps even");
    if (nout < 0 || !(step > 0) || pos0 < 0)
        throw std::invalid_argument("polyphase_read: bad position, step or count");
    const int taps = static_cast<int>(table_a.shape(1)), nph = static_cast<int>(table_a.shape(0)) - 1, half = taps/2;
    const long long S = buf_a.size();
    // The first output reads from i0 - half + 1 >= 0 and the last up to i0 + half <= S - 1
    if (nout > 0 && (static_cast<long long>(pos0) - half + 1 < 0 || static_cast<long long>(pos0 + static_cast<double>(nout - 1)*step) + half > S - 1))
        throw std::invalid_argument("polyphase_read: the kernels reach outside the buffer");
    py::array_t<E> out(static_cast<py::ssize_t>(nout));
    const float *buf = reinterpret_cast<const float *>(buf_a.data());
    std::vector<float> doubled; // the complex read wants each tap twice (see read_complex)
    const float *tab = table_a.data();
    if constexpr (std::is_same_v<E, cf32>) {
        doubled.resize(2*static_cast<std::size_t>(table_a.size()));
        for (py::ssize_t i = 0; i < table_a.size(); i++) doubled[2*i] = doubled[2*i + 1] = tab[i];
        tab = doubled.data();
    }
    float *o = reinterpret_cast<float *>(out.mutable_data());
    constexpr long long CHUNK = 4096;
    {
        py::gil_scoped_release nogil;
        parallel_for((nout + CHUNK - 1)/CHUNK, threads, [&](long long blk) {
            if constexpr (std::is_same_v<E, cf32>) read_complex(buf, tab, taps, nph, pos0, step, blk*CHUNK, std::min(nout, (blk + 1)*CHUNK), o);
            else read_real(buf, tab, taps, nph, pos0, step, blk*CHUNK, std::min(nout, (blk + 1)*CHUNK), o);
        });
    }
    return out;
}

double rotate_into(carray<cf32> x_a, py::array_t<cf32, py::array::c_style> out_a, double phase0, double w, int threads) {
    const long long n = x_a.size();
    if (out_a.size() != n || !out_a.writeable())
        throw std::invalid_argument("rotate_into: out must be a writeable array as long as x");
    const cf32 *x = x_a.data();
    cf32 *o = out_a.mutable_data();
    constexpr long long CHUNK = 8192;
    constexpr double TWO_PI = 6.283185307179586;
    {
        py::gil_scoped_release nogil;
        parallel_for((n + CHUNK - 1)/CHUNK, threads, [&](long long blk) {
            const long long a = blk*CHUNK, b = std::min(n, a + CHUNK);
            const double ph = std::fmod(phase0 + w*static_cast<double>(a), TWO_PI); // the chunk's own start: no drift between chunks
            double pr = std::cos(ph), pi = std::sin(ph);
            const double rr = std::cos(w), ri = std::sin(w);
            for (long long i = a; i < b; i++) {
                const float xr = x[i].real(), xi = x[i].imag();
                o[i] = cf32(static_cast<float>(xr*pr - xi*pi), static_cast<float>(xr*pi + xi*pr));
                const double nr = pr*rr - pi*ri, ni = pr*ri + pi*rr;
                pr = nr; pi = ni;
            }
        });
    }
    double end = std::fmod(phase0 + w*static_cast<double>(n), TWO_PI);
    return end < 0 ? end + TWO_PI : end; // as Python's %, in [0, 2 pi)
}

} // namespace

void register_resample(py::module_ &m) {
    const char *doc = "polyphase_read(buf, table float32 (phases + 1, taps), pos0, step, count, threads) -> out[k] = sum_j table[nearest phase][j] buf[floor(pos0 + k step) - taps/2 + 1 + j]";
    m.def("polyphase_read", &polyphase_read<float>, doc, py::arg("buf").noconvert(), py::arg("table"), py::arg("pos0"), py::arg("step"), py::arg("count"), py::arg("threads") = 0);
    m.def("polyphase_read", &polyphase_read<cf32>, doc, py::arg("buf").noconvert(), py::arg("table"), py::arg("pos0"), py::arg("step"), py::arg("count"), py::arg("threads") = 0);
    m.def("rotate_into", &rotate_into, "rotate_into(x complex64, out complex64, phase0, w, threads) -> phase after len(x): out = x exp(j (phase0 + w i))",
          py::arg("x"), py::arg("out").noconvert(), py::arg("phase0"), py::arg("w"), py::arg("threads") = 0);
}

} // namespace rpps
