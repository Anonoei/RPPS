from types import SimpleNamespace

import numpy as np

from ...base import Stage
from ...base.kernel import device_functions, has_device_kernels, to_host
from .. import track
from ... import _accel, _effort
from ...math import interp

R = interp.REACH # Samples the symbol interpolation reads on each side (any effort level's: one may change mid-stream)

# On a device: the held samples go up once a call; each window's square-law sums are one reduction (blocks of partial
# sums in a fixed order, added on the host in double: the line read as the host makes it, rpps.sync.track.line_parts),
# and its symbols one launch reading the samples at instants in 32.32 fixed point (made on the host in double: consumer
# GPUs run float64 slowly), cubic or by the windowed-sinc table, zeros outside what is held (as interp.fractional_edge).
OM_SOURCE = r"""
__device__ __forceinline__ float2 cmul(const float2 a, const float2 b) {
    return make_float2(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x);
}
// Partial sums, a block each, of p = |x[off + i]|^2 line[i] over i < h and h <= i < n, and of |x|^2
extern "C" __global__ void om_sum(const float2 *x, const long long off, const long long n, const long long h,
                                  const float2 *starts, const float2 *local, const int B, float *part) {
    __shared__ float s[5][256];
    float a[5] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (long long i = (long long)blockIdx.x*blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x*blockDim.x) {
        const float2 v = x[off + i];
        const float e = v.x*v.x + v.y*v.y;
        const float2 p = cmul(make_float2(e, 0.0f), cmul(starts[i/B], local[i % B]));
        const int k = i < h ? 0 : 2;
        a[k] += p.x; a[k + 1] += p.y; a[4] += e;
    }
    for (int k = 0; k < 5; k++) s[k][threadIdx.x] = a[k];
    __syncthreads();
    for (int w = blockDim.x/2; w > 0; w >>= 1) {
        if (threadIdx.x < w) for (int k = 0; k < 5; k++) s[k][threadIdx.x] += s[k][threadIdx.x + w];
        __syncthreads();
    }
    if (threadIdx.x == 0) for (int k = 0; k < 5; k++) part[5*blockIdx.x + k] = s[k][0];
}
// x at instants A + k S (32.32 fixed point, relative to x[0]), k < count: cubic (taps 4) or the windowed-sinc table
// (phases + 1 rows of taps, blended linearly, reading x[i + n0 .. i + n0 + taps)); outside x: zeros
extern "C" __global__ void om_values(const float2 *x, const long long nx, const long long A, const long long S,
                                     const long long count, const int taps, const float *tab, const int phases,
                                     const int n0, float2 *out) {
    const long long k = (long long)blockIdx.x*blockDim.x + threadIdx.x;
    if (k >= count) return;
    const long long t = A + k*S, i = t >> 32;
    const float u = (float)(unsigned int)(t & 0xffffffffLL)*2.3283064365386963e-10f;
    float sr = 0.0f, si = 0.0f;
    if (taps == 4) {
        const float c[4] = {-u*(u - 1.0f)*(u - 2.0f)/6.0f, (u + 1.0f)*(u - 1.0f)*(u - 2.0f)/2.0f,
                            -(u + 1.0f)*u*(u - 2.0f)/2.0f, (u + 1.0f)*u*(u - 1.0f)/6.0f};
        for (int m = 0; m < 4; m++) {
            const long long q = i - 1 + m;
            if (q < 0 || q >= nx) continue;
            sr += c[m]*x[q].x; si += c[m]*x[q].y;
        }
    } else {
        const float p = u*phases;
        const int r = min((int)floorf(p), phases - 1);
        const float b = p - r;
        for (int m = 0; m < taps; m++) {
            const long long q = i + n0 + m;
            if (q < 0 || q >= nx) continue;
            const float h = (1.0f - b)*tab[r*taps + m] + b*tab[(r + 1)*taps + m];
            sr += h*x[q].x; si += h*x[q].y;
        }
    }
    out[k] = make_float2(sr, si);
}
"""

class OerderMeyr(Stage):
    """Symbol timing by square-law block estimation (Oerder & Meyr), tracked across windows: samples at `sps` in (after
    their matched filter), one symbol each out

        symbols = matched + rp.sync.time.OerderMeyr()

    First class (rpps.sync). Windows sit at absolute stream positions [k*W, (k+1)*W): W is `window`
    seconds, at least `min_symbols` symbols (the square-law line is weak: at roll-off 0.2 it is about 3% of the power, so it
    wants a few thousand samples). Each window's timing phase is the angle of sum |x[n]|^2 exp(-j 2 pi n / sps), and a
    block-rate loop (rpps.sync.track) carries the phase and the clock's rate (ppm: the transmitter's clock fast by that much) from window to window, so a clock some ppm
    off is followed and each symbol's instant comes from the line, not a step per window. A window whose line is weaker
    than `gate` (its SNR, |X| / sum|x|^2 * sqrt(n): noise alone is about 1) is skipped, the loop coasting through it (a gap
    between bursts). Symbols are interpolated from the samples (cubic, or a windowed sinc under 4 samples a symbol:
    rpps._effort.interp_taps); any sps from 1, fractional too (under 2: a 30 MBd carrier in 50 MS/s), and any block cut.

    Latency: a window, as `latency` (seconds) in stream.est[name] with the phase, the rate (ppm) and the line's SNR.
    Nothing comes out before the first window that passes the gate. `device`: each window's sums and symbols on the GPU
    of `rpps._accel.device()` (CuPy, CUDA or ROCm/HIP; the host without one), the samples going up once a block; the
    tracking is the host's, window by window (under 4 samples a symbol the upsampled window is made on the host).
    """
    CONTROLS = {}
    ACQUIRE = 4 # Windows the loop runs wide (bandwidth 0.6) while it pulls in, then `bw`

    THREADS, BLOCKS = 256, 32 # A window's sums: BLOCKS partial sums come back

    def __init__(self, sps=None, window=1e-3, min_symbols=2048, bw=0.25, gate=2.0, name="timing", clock=True, device=False):
        self.sps, self.window, self.min_symbols, self.bw, self.gate, self.name = sps, window, min_symbols, bw, gate, name
        self.device = bool(device)
        self.clock = bool(clock) # False: the phase only, no clock rate (a burst a few windows long: two windows' phases
        # say nothing of the clock, and a rate taken from them slips symbols)

    def __str__(self):
        return "OerderMeyr"

    def new_state(self, meta):
        xp = _accel.device()[0] if self.device else np
        return SimpleNamespace(buf=np.zeros(0, np.complex64), base=None, k=None, W=None, m=None, tr=track.new(), snr=0.0,
                               xp=xp, dx=None)

    def _dev(self, st):
        """True when this call runs on a device"""
        return self.device and has_device_kernels(st.xp)

    def _held(self, st):
        """(xp, functions, the held samples on the device): up once a call"""
        xp = st.xp
        f = device_functions(xp, OM_SOURCE, ("om_sum", "om_values"))
        if st.dx is None:
            st.dx = xp.asarray(np.ascontiguousarray(st.buf, dtype=np.complex64))
        return xp, f, st.dx

    def _sums(self, st, seg, a0, u, sps):
        """(X, sum of the first half, of the second, sum |x|^2) of p = |x|^2 e^{-j 2 pi n / (sps u)} over the window"""
        line = (2*np.pi*a0/sps, 2*np.pi/(sps*u))
        h = len(seg)//2
        if not self._dev(st) and len(seg) and _accel.native is not None: # One pass in C++ (the lines below are its reference)
            first, second, energy = _accel.native.om_sums(np.ascontiguousarray(seg), *line, h)
            return first + second, first, second, float(energy)
        if not self._dev(st) or not len(seg):
            p = (seg.real**2 + seg.imag**2)*track.line_phasor(*line, len(seg)) # |x|^2 e^{-j 2 pi n / sps}
            first, second = np.sum(p[:h]), np.sum(p[h:])
            return np.sum(p), first, second, float(np.sum(np.abs(seg)**2))
        xp, f, dx = self._held(st)
        if u == 1:
            src, off = dx, np.int64(a0 - st.base)
        else: # (the upsampled window, made on the host)
            src, off = xp.asarray(np.ascontiguousarray(seg, dtype=np.complex64)), np.int64(0)
        starts, local = track.line_parts(*line, len(seg))
        part = xp.empty(5*self.BLOCKS, dtype=np.float32)
        f["om_sum"]((self.BLOCKS,), (self.THREADS,), (src, off, np.int64(len(seg)), np.int64(h),
                                                      xp.asarray(starts.astype(np.complex64)),
                                                      xp.asarray(local.astype(np.complex64)), np.int32(len(local)), part))
        s = to_host(part).astype(np.float64).reshape(-1, 5).sum(axis=0)
        first, second = complex(s[0], s[1]), complex(s[2], s[3])
        return first + second, first, second, float(s[4])

    def step(self, x, st, meta):
        sps = float(self.sps or meta.sps)
        if sps < 1:
            raise ValueError(f"OerderMeyr needs a sample a symbol or more, got {sps}")
        out = []
        st.dx = None # (the held samples on the device: up again once this call's are in)
        W = max(int(round(self.window*meta.Fs)), int(np.ceil(self.min_symbols*sps)))
        for a0, b, seg in track.windows(st, x, meta, W):
            end = st.base + len(st.buf)
            if len(seg) >= 16*sps:
                # Under 4 samples a symbol |x|^2 reaches past Nyquist and its line at 1/T aliases onto its mirror (at 2 they
                # cancel at some phases): the line is taken at u times the rate, u the least power of 2 to 4 samples a symbol
                # (under 2, a 30 MBd carrier in 50 MS/s: the samples hold the signal, its square wants more of them)
                u = 1 if sps >= 4 else 1 << int(np.ceil(np.log2(4/sps)))
                if u == 1:
                    if _accel.native is None or self._dev(st): # (C++ sums single precision samples in double as they are)
                        seg = seg.astype(np.complex128) # (held as it came, single precision; the line is summed in double)
                else: # x at 1/u samples, band limited, from what the window holds (so any block cut gives the same)
                    seg = interp.upsampled(st.buf, a0, b, st.base, u)
                X, first, second, energy = self._sums(st, seg, a0, u, sps)
                st.snr = float(np.abs(X)/max(energy, 1e-30)*np.sqrt(len(seg)/u))
                if st.snr >= self.gate:
                    tau = (-np.angle(X)*sps/(2*np.pi)) % sps
                    rate = None
                    if not st.tr.started: # The clock's rate to start from: the two halves of the window
                        h = len(seg)//2
                        if abs(first) > 0 and abs(second) > 0: # (a half of zeros, as before a stream starts, says nothing)
                            rate = track.wrap(-np.angle(second/first)*sps/(2*np.pi), sps)/(h/u)
                    bw = self.bw if st.tr.windows >= self.ACQUIRE else max(self.bw, 0.6) # Wide while it pulls in
                    track.update(st.tr, tau, (a0 + b)/2, sps, bw, rate if self.clock else 0.0)
                    if not self.clock:
                        st.tr.rate = 0.0
                else:
                    track.miss(st.tr, (a0 + b)/2)
            if st.tr.started: # The first time from what is held: the window before, too (a signal that starts part way into a
                # window gated out has its first symbols there: a frame's preamble)
                start = a0 if st.m is not None else st.base
                out.append(self._symbols(st, start, min(b, end - (3 if meta.flush else R + 1)), sps)) # (flushed: zeros after the end)
        # Keep the window being gathered (until the tracker starts, the one before it too), and what the next symbol's
        # interpolation needs
        keep = st.k*st.W if st.tr.started else (st.k - 1)*st.W
        if st.m is not None:
            keep = min(keep, int(np.floor(self._time(st, st.m, sps))) - R)
        track.keep(st, keep)
        W, tr = st.W, st.tr
        meta.stream.est[self.name] = dict(phase=tr.value % sps if tr.started else None, ppm=-tr.rate*1e6 if tr.started else None,
                                          snr=st.snr, windows=tr.windows, missed=tr.missed, latency=W/meta.Fs)
        meta.op.Fs = meta.Fs/sps
        meta.op.sps = 1
        return np.concatenate(out).astype(np.complex64) if out else np.zeros(0, np.complex64)

    @staticmethod
    def _time(st, m, sps):
        """Instant of symbol m: t = theta(t) + m*sps, theta the tracked line"""
        tr = st.tr
        return (m*sps + tr.value - tr.rate*tr.at)/(1 - tr.rate)

    def _symbols(self, st, a, b, sps):
        """The symbols whose instants fall before b (the window's end, or what the interpolation can reach)"""
        tr = st.tr
        if st.m is None:
            st.m = int(np.ceil((a + 1 - track.at(tr, a))/sps))
        hi = (b*(1 - tr.rate) - tr.value + tr.rate*tr.at)/sps # m*sps < b(1 - r) - v + r*c
        m = np.arange(st.m, max(int(np.ceil(hi)), st.m))
        if not len(m):
            return np.zeros(0, np.complex128)
        t = self._time(st, m, sps)
        ok = t - 1 >= st.base # (only before the first window, never after; samples before the stream are zeros)
        m, t = m[ok], t[ok]
        st.m = int(m[-1]) + 1 if len(m) else st.m
        if not len(t):
            return np.zeros(0, np.complex128)
        taps = _effort.interp_taps(sps)
        if not self._dev(st):
            return interp.fractional_edge(st.buf, t - st.base, taps)
        xp, f, dx = self._held(st)
        fix = lambda v: np.int64(np.round(v*4294967296.0))
        A, S = fix(self._time(st, m[0], sps) - st.base), fix(sps/(1 - tr.rate)) # (t is a line in m: its first and step)
        tab, n = interp._sinc_table(taps) if taps != 4 else (np.zeros((1, 1)), np.zeros(1, np.int64))
        dtab = self._table(xp, taps, tab)
        out = xp.empty(len(t), dtype=np.complex64)
        T = self.THREADS
        f["om_values"](((len(t) + T - 1)//T,), (T,), (dx, np.int64(len(dx)), A, S, np.int64(len(t)), np.int32(taps), dtab,
                                                       np.int32(interp._PHASES), np.int32(n[0]), out))
        return to_host(out)

    _tables = {}

    @classmethod
    def _table(cls, xp, taps, tab):
        """The windowed-sinc table on the device, once per array module and taps"""
        key = (getattr(xp, "__name__", id(xp)), taps)
        if key not in cls._tables:
            cls._tables[key] = xp.asarray(np.ascontiguousarray(tab, dtype=np.float32))
        return cls._tables[key]
