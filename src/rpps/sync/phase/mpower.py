from types import SimpleNamespace

import numpy as np

from ... import _accel
from ...base import Stage
from ...base.kernel import device_functions, has_device_kernels, to_host
from .. import track
from ...math.interp import peak

# On a device: the held samples go up once a call; each window's measurement is one reduction (blocks of partial sums in
# a fixed order, added on the host in double) and its correction one launch, the phase line read as the host makes it
# (rpps.sync.track.line_parts: exponentials in double, a product of two an item). Portable CUDA C.
MP_SOURCE = r"""
__device__ __forceinline__ float2 cmul(const float2 a, const float2 b) {
    return make_float2(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x);
}
// Partial sums, a block each, of z = unit(x[off + i] line[i])^M and |z| over i < n
extern "C" __global__ void mp_sum(const float2 *x, const long long off, const long long n, const float2 *starts,
                                  const float2 *local, const int B, const int M, float *part) {
    __shared__ float pr[256], pi[256], pm[256];
    float ar = 0.0f, ai = 0.0f, am = 0.0f;
    for (long long i = (long long)blockIdx.x*blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x*blockDim.x) {
        const float2 y = cmul(x[off + i], cmul(starts[i/B], local[i % B]));
        const float a = sqrtf(y.x*y.x + y.y*y.y), inv = 1.0f/fmaxf(a, 1e-30f);
        float ur = y.x*inv, ui = y.y*inv, rr = 1.0f, ri = 0.0f, um = a*inv, rm = 1.0f;
        for (int m = M; m; m >>= 1) {
            if (m & 1) { const float t = rr*ur - ri*ui; ri = rr*ui + ri*ur; rr = t; rm *= um; }
            const float t = ur*ur - ui*ui; ui = 2.0f*ur*ui; ur = t; um *= um;
        }
        ar += rr; ai += ri; am += rm;
    }
    pr[threadIdx.x] = ar; pi[threadIdx.x] = ai; pm[threadIdx.x] = am;
    __syncthreads();
    for (int w = blockDim.x/2; w > 0; w >>= 1) {
        if (threadIdx.x < w) {
            pr[threadIdx.x] += pr[threadIdx.x + w]; pi[threadIdx.x] += pi[threadIdx.x + w]; pm[threadIdx.x] += pm[threadIdx.x + w];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) { part[3*blockIdx.x] = pr[0]; part[3*blockIdx.x + 1] = pi[0]; part[3*blockIdx.x + 2] = pm[0]; }
}
// x[off + i] turned back by the line, i < n
extern "C" __global__ void mp_turn(const float2 *x, const long long off, const long long n, const float2 *starts,
                                   const float2 *local, const int B, float2 *out) {
    const long long i = (long long)blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) out[i] = cmul(x[off + i], cmul(starts[i/B], local[i % B]));
}
"""

class MPower(Stage):
    """Carrier frequency and phase of M-fold symmetric symbols by the M-th power, block estimation tracked across windows

        symbols = timed + rp.sync.phase.MPower(order=4, offset=np.pi)     # QPSK at 45 degrees: its points^4 are -1

    First class (rpps.sync). Windows sit at absolute stream positions, `window` seconds (at least
    `min_symbols`). The first window that shows the carrier acquires it: the FFT of y^M (zero padded) gives the
    frequency, |f| < 1/(2M) cycles per symbol, and the phase at its centre. After that each window measures the
    residual phase of y turned back by the tracked line, (angle(sum u^M) - offset) / M with u = y e^{-j phase} / |y|, and a block-rate
    loop (rpps.sync.track) carries phase and frequency on, so the correction is a line per symbol, with no staircase and no
    slips while the drift stays inside the loop. A window whose M-th power line is weaker than `gate` (its SNR: noise
    alone is about 1) is skipped; after `relock` of those in a row the carrier is acquired again from the FFT. Acquiring
    wants `gate` more than the best of the FFT's N frequencies reaches on noise, sqrt(2 ln N): at an Es/N0 where the M-th
    power is lost in noise (DVB-S2 under 0 dB with M = 8) nothing is acquired and the symbols pass, for a preamble and
    pilots to take the phase.

    The phase is known modulo 2 pi / M: a preamble, a differential code or a polarity search takes the rest. order: M (a
    profile may change it with the modulation, live). offset: the angle of the constellation's mean point^M (0 for BPSK,
    8PSK, and DVB-S2's pi/2-BPSK under M = 8; pi for QPSK at 45 degrees under M = 4).
    Until it acquires, symbols pass unchanged. Estimates in stream.est[name]: frequency (cycles per symbol), phase, SNR,
    latency (seconds). `device`: each window's measurement and correction on the GPU of `rpps._accel.device()` (CuPy,
    CUDA or ROCm/HIP; the host without one), the samples going up once a block; acquiring (an FFT, rarely) stays on the
    host, and the tracking is the host's, window by window.
    """
    CONTROLS = {"order": "live"}

    THREADS, BLOCKS = 256, 32 # A measurement's launch: BLOCKS partial sums come back

    def __init__(self, order=4, offset=0.0, window=1e-3, min_symbols=1024, bw=0.25, gate=3.0, relock=4, name="carrier",
                 device=False):
        self.order, self.offset, self.window, self.min_symbols = int(order), float(offset), window, min_symbols
        self.bw, self.gate, self.relock, self.name, self.device = bw, gate, relock, name, bool(device)

    def __str__(self):
        return f"MPower({self.order})"

    def control(self, order=None):
        if order is not None:
            self.order = int(order)

    def pos_rate(self, decode):
        return 1.0

    def new_state(self, meta):
        xp = _accel.device()[0] if self.device else np
        return SimpleNamespace(buf=np.zeros(0, np.complex64), base=None, k=None, W=None, tr=track.new(), snr=0.0, xp=xp,
                               dx=None)

    def _acquire(self, seg, n, c):
        M = self.order
        z = _power(_unit(seg.astype(np.complex128)), M)
        N = 4*(1 << int(np.ceil(np.log2(max(len(z), 2)))))
        F = np.abs(np.fft.fft(z, N))
        i, d = peak(F, circular=True, log=False)
        f = (((i + d)/N + 0.5) % 1.0 - 0.5)/M # cycles per symbol
        s = np.sum(z*np.exp(-2j*np.pi*M*f*(n - c)))
        return f, s, z

    def step(self, x, st, meta):
        M, out = self.order, []
        st.dx = None # (the held samples on the device: up again once this call's are in)
        for a0, b, seg in track.windows(st, x, meta, max(int(round(self.window*meta.Fs)), int(self.min_symbols))):
            n = np.arange(a0, b)
            c = (a0 + b)/2
            tr = st.tr
            if len(seg) >= 8:
                if not tr.started or tr.missed >= self.relock:
                    f, s, z = self._acquire(seg, n, c)
                    st.snr = float(np.abs(s)/max(np.sum(np.abs(z)), 1e-30)*np.sqrt(len(seg)))
                    # The best of N frequencies: noise alone reaches about sqrt(2 ln N) there, so acquiring wants more
                    N = 4*(1 << int(np.ceil(np.log2(max(len(z), 2)))))
                    if st.snr >= self.gate + np.sqrt(2*np.log(N)):
                        tr.started = False
                        track.update(tr, (np.angle(s) - self.offset)/M, c, rate=2*np.pi*f)
                else:
                    s, mag = self._measure(st, a0, seg, track.at(tr, a0), tr.rate, M)
                    st.snr = float(np.abs(s)/max(mag, 1e-30)*np.sqrt(len(seg)))
                    if st.snr >= self.gate:
                        e = track.wrap((np.angle(s) - self.offset)/M, 2*np.pi/M)
                        bw = self.bw if tr.windows >= 4 else max(self.bw, 0.6) # Wide while it pulls in
                        track.update(tr, track.at(tr, c) + e, c, 2*np.pi/M, bw)
                    else:
                        track.miss(tr, c)
            out.append(self._turn_back(st, a0, seg, track.at(tr, a0), tr.rate) if tr.started else seg.astype(np.complex64))
        track.keep(st, st.k*st.W)
        W, tr = st.W, st.tr
        meta.stream.est[self.name] = dict(frequency=tr.rate/(2*np.pi) if tr.started else None,
                                          phase=float(tr.value % (2*np.pi/M)) if tr.started else None, snr=st.snr,
                                          windows=tr.windows, missed=tr.missed, latency=W/meta.Fs)
        return np.concatenate(out).astype(np.complex64) if out else np.zeros(0, np.complex64)

    # ---- a window's measurement and correction: on the host (C++ or numpy), or on a device

    def _on_device(self, st, a0, n, phi0, rate):
        """(xp, functions, the window's offset in st.dx, starts, local, B): the held samples up once a call"""
        xp = st.xp
        f = device_functions(xp, MP_SOURCE, ("mp_sum", "mp_turn"))
        if st.dx is None:
            st.dx = xp.asarray(np.ascontiguousarray(st.buf, dtype=np.complex64))
        starts, local = track.line_parts(phi0, rate, n)
        return xp, f, np.int64(a0 - st.base), xp.asarray(starts.astype(np.complex64)), xp.asarray(local.astype(np.complex64)), np.int32(len(local))

    def _measure(self, st, a0, seg, phi0, rate, M):
        if not (self.device and has_device_kernels(st.xp)) or not len(seg):
            return _measure(seg, phi0, rate, M)
        xp, f, off, starts, local, B = self._on_device(st, a0, len(seg), phi0, rate)
        part = xp.empty(3*self.BLOCKS, dtype=np.float32)
        f["mp_sum"]((self.BLOCKS,), (self.THREADS,), (st.dx, off, np.int64(len(seg)), starts, local, B, np.int32(M), part))
        part = to_host(part).astype(np.float64).reshape(-1, 3).sum(axis=0)
        return complex(part[0], part[1]), float(part[2])

    def _turn_back(self, st, a0, seg, phi0, rate):
        if not (self.device and has_device_kernels(st.xp)) or not len(seg):
            return _turn_back(seg, phi0, rate)
        xp, f, off, starts, local, B = self._on_device(st, a0, len(seg), phi0, rate)
        out = xp.empty(len(seg), dtype=np.complex64)
        T = self.THREADS
        f["mp_turn"](((len(seg) + T - 1)//T,), (T,), (st.dx, off, np.int64(len(seg)), starts, local, B, out))
        return to_host(out)

def _measure(seg, phi0, rate, M):
    """(sum z, sum |z|) of z = unit(seg turned back by the phase line)^M: a window's measurement (one C++ call, or numpy)"""
    if _accel.native is not None:
        return _accel.native.mpower_sum(seg, phi0, rate, M)
    z = _power(_unit(seg*track.line_phasor(phi0, rate, len(seg))), M)
    return np.sum(z), float(np.sum(np.abs(z)))

def _turn_back(seg, phi0, rate):
    """seg turned back by the phase line, single precision"""
    if _accel.native is not None:
        return _accel.native.rotate_line(seg, phi0, rate)
    return (seg*track.line_phasor(phi0, rate, len(seg))).astype(np.complex64)

def _unit(y):
    """y with its magnitudes taken off: the phase alone goes into the M-th power (Viterbi & Viterbi with no amplitude
    weight), so a few large noise samples can't make a line, and noise alone gives |sum z| / sqrt(n) about 1"""
    return y/np.maximum(np.abs(y), 1e-30) # (a zero stays zero)

def _power(u, M):
    """u**M for an integer M, by squaring: a complex power is several times slower than the multiplications"""
    out, base = None, u
    while M:
        if M & 1:
            out = base if out is None else out*base
        M >>= 1
        if M:
            base = base*base
    return out if out is not None else np.ones_like(u)

def offset_of(constellation, order):
    """The angle of a constellation's mean point^M: what MPower takes off as `offset`"""
    pts = np.asarray(constellation.points)
    return float(np.angle(np.mean(pts**order)))
