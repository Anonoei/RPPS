"""MatTrack: the matched filter and symbol timing in one, tracked across windows (a profile's `timing: mat`)"""
from types import SimpleNamespace

import numpy as np

from ... import _accel, _effort
from ...base import Stage
from ...base.kernel import device_functions, has_device_kernels, to_host
from ...math import interp
from ...math.interp import peak
from ...signal.filters.convolve import convolve
from .. import track

R = interp.REACH

# On a device: the held samples filtered once by a tiled FIR (a block's outputs read one shared tile of samples, so
# neighbouring threads read neighbouring samples), then y read at the instants, cubic, four samples each: the same values
# as the reference (numpy: filter, then `fractional`). Instants come as sets: the first (A) and the step (S) of each, in
# 32.32 fixed point, made on the host in double (consumer GPUs run float64 slowly). Portable CUDA C.
MF_SOURCE = r"""
#define MF_TILE 256
// y[n] = sum_k h[k] x[n - k], x before its start zeros. Dynamic shared memory: the tile (MF_TILE + nh - 1 samples) and
// the taps
extern "C" __global__ void mf_fir(const float2 *x, const long long nx, const float2 *h, const int nh, const int cplx,
                                  float2 *y) {
    extern __shared__ float2 sm[];
    float2 *tile = sm, *taps = sm + MF_TILE + nh - 1;
    const long long n0 = (long long)blockIdx.x*MF_TILE;
    for (int k = threadIdx.x; k < nh; k += blockDim.x) taps[k] = h[k];
    for (int j = threadIdx.x; j < MF_TILE + nh - 1; j += blockDim.x) {
        const long long q = n0 - (nh - 1) + j;
        tile[j] = (q >= 0 && q < nx) ? x[q] : make_float2(0.0f, 0.0f);
    }
    __syncthreads();
    const long long n = n0 + threadIdx.x;
    if (threadIdx.x >= MF_TILE || n >= nx) return;
    float sr = 0.0f, si = 0.0f;
    const int j0 = threadIdx.x + nh - 1; // tile index of x[n]
    if (cplx) {
        for (int k = 0; k < nh; k++) {
            const float2 v = tile[j0 - k], c = taps[k];
            sr += c.x*v.x - c.y*v.y; si += c.x*v.y + c.y*v.x;
        }
    } else {
        for (int k = 0; k < nh; k++) {
            const float2 v = tile[j0 - k];
            const float c = taps[k].x;
            sr += c*v.x; si += c*v.y;
        }
    }
    y[n] = make_float2(sr, si);
}
// y at a fixed-point instant t, cubic through y[i - 1 .. i + 2] (outside y: zeros)
__device__ float2 cubic(const float2 *y, const long long ny, const long long t) {
    const long long i = t >> 32;
    const float u = (float)(unsigned int)(t & 0xffffffffLL)*2.3283064365386963e-10f;
    const float c[4] = {-u*(u - 1.0f)*(u - 2.0f)/6.0f, (u + 1.0f)*(u - 1.0f)*(u - 2.0f)/2.0f,
                        -(u + 1.0f)*u*(u - 2.0f)/2.0f, (u + 1.0f)*u*(u - 1.0f)/6.0f};
    float sr = 0.0f, si = 0.0f;
    for (int m = 0; m < 4; m++) {
        const long long q = i - 1 + m;
        if (q < 0 || q >= ny) continue;
        const float2 v = y[q];
        sr += c[m]*v.x; si += c[m]*v.y;
    }
    return make_float2(sr, si);
}
// Sum of |y|^2 over each set: a block a set, its threads over the items, reduced in shared memory (a fixed order: the same
// sums whatever else is in the launch)
extern "C" __global__ void mf_energy(const float2 *y, const long long ny, const long long *A, const long long *S,
                                     const int *cnt, float *sums) {
    __shared__ float part[256];
    const int s = blockIdx.x;
    float acc = 0.0f;
    for (long long k = threadIdx.x; k < cnt[s]; k += blockDim.x) {
        const float2 v = cubic(y, ny, A[s] + k*S[s]);
        acc += v.x*v.x + v.y*v.y;
    }
    part[threadIdx.x] = acc;
    __syncthreads();
    for (int w = blockDim.x/2; w > 0; w >>= 1) {
        if (threadIdx.x < w) part[threadIdx.x] += part[threadIdx.x + w];
        __syncthreads();
    }
    if (threadIdx.x == 0) sums[s] = part[0];
}
// y at every item of the sets, one after another (start: each set's first output index, nsets + 1 of them)
extern "C" __global__ void mf_values(const float2 *y, const long long ny, const long long *A, const long long *S,
                                     const long long *start, const int nsets, float2 *out) {
    const long long t = (long long)blockIdx.x*blockDim.x + threadIdx.x;
    if (t >= start[nsets]) return;
    int a = 0, b = nsets - 1; // (the set of item t)
    while (a < b) { const int mid = (a + b + 1)/2; if (start[mid] <= t) a = mid; else b = mid - 1; }
    out[t] = cubic(y, ny, A[a] + (t - start[a])*S[a]);
}
"""

class MatTrack(Stage):
    """Matched filter and symbol timing in one: samples at `sps` in (before their matched filter, which a profile fuses in
    from the step before: `fuse`), one symbol each out. The matched filter is evaluated only at the instants read, a dot
    product of its taps with the samples at the four around each instant, cubic-interpolated: what filtering every sample
    and interpolating (Matched + OerderMeyr) gives there, `sps` times less filtering.

        symbols = samples + rp.sync.time.MatTrack(sps=4, pulse=rrc)           # or a profile's {"timing": {"method": "mat"}}

    The timing is a matrix matched filter's (as `Mat`, MatRRC): no decisions, no per-sample loop. Each window (`window`
    seconds, at least `min_symbols` symbols, at absolute stream positions) measures the filter's energy at the predicted
    instants and `delta` samples either side; the parabola's vertex is the timing error, and the block-rate loop
    (rpps.sync.track) carries the phase and the clock's rate on. The first window acquires the phase from `grid` phases a
    symbol (the energy's maximum), as does a window after `relock` misses; a window without a maximum is a miss (noise,
    a gap between bursts), the loop coasting through it.

    Once the loop has settled (`group` windows), windows go in groups of `group` at fixed stream positions, each measured at
    the instants the loop predicted at its group's start, so a group is one batch of work (one launch on a device) and any
    block cut gives the same symbols; `delta` (default a quarter symbol) sets the range a measurement reads, which a
    settled loop stays well inside of over a group. `device`: the evaluation on the
    GPU of `rpps._accel.device()` (CuPy, CUDA or ROCm/HIP; numpy without one): only the samples go up and the symbols
    down. Estimates in stream.est[name]: phase (samples), ppm, windows, missed, latency (seconds)."""
    CONTROLS = {}
    ACQUIRE = 4 # Windows the loop runs wide (bandwidth 0.6) and measures one at a time while it pulls in
    THREADS = 256

    def __init__(self, sps=None, window=1e-3, min_symbols=2048, bw=0.25, delta=None, grid=16, gate=0.02, relock=4, group=16,
                 device=False, name="timing"):
        self.sps, self.window, self.min_symbols, self.bw, self.delta = sps, window, min_symbols, bw, delta
        self.grid, self.gate, self.relock, self.group, self.device, self.name = int(grid), gate, int(relock), int(group), bool(device), name
        self.matched = None

    def fuse(self, matched):
        """Take over a matched filter (a FIR stage, the profile's `wave shape` receive side): it is evaluated here"""
        self.matched = matched

    def __str__(self):
        return "MatTrack" if self.matched is None else "MatTrack(matched)"

    def new_state(self, meta):
        taps = np.ones(1) if self.matched is None else np.asarray(self.matched.new_state(meta).taps)
        xp = _accel.device()[0] if self.device else np
        return SimpleNamespace(buf=np.zeros(0, np.complex64), base=None, k=None, W=None, m=None, tr=track.new(), taps=taps,
                               xp=xp, snap=None, gidx=None, contrast=0.0, y=None, dy=None, dtaps=None)

    # ---- the matched filter at fractional instants: sets of them, (m0, count, (v, r, at, off)), instant k of a set at
    # ((m0 + k) sps + v - r at)/(1 - r) + off (a tracked line's symbols, moved by off)

    @staticmethod
    def _instants_of(sets, sps):
        return [((np.arange(m0, m0 + n)*sps + v - r*at)/(1 - r) + off) for m0, n, (v, r, at, off) in sets]

    def _reference(self, st, t):
        """y at absolute instants t: every sample held filtered, then read where wanted (numpy: the reference)"""
        if st.y is None:
            st.y = convolve(st.buf.astype(np.complex128), st.taps)[:len(st.buf)] # (by FFT where it pays)
        return interp.fractional_edge(st.y, np.asarray(t, dtype=np.float64) - st.base, 4) if len(t) else np.zeros(0, np.complex128)

    def _device(self, st):
        """(xp, functions), the held samples filtered on the device once a call (st.dy)"""
        xp = st.xp
        f = device_functions(xp, MF_SOURCE, ("mf_fir", "mf_energy", "mf_values"))
        if st.dy is None:
            if st.dtaps is None:
                st.dtaps = xp.asarray(np.ascontiguousarray(st.taps, dtype=np.complex64))
            x = xp.asarray(np.ascontiguousarray(st.buf, dtype=np.complex64))
            st.dy = xp.empty_like(x)
            nh, tile = len(st.taps), 256
            if len(x):
                f["mf_fir"](((len(x) + tile - 1)//tile,), (tile,), (x, np.int64(len(x)), st.dtaps, np.int32(nh), self._cplx(st), st.dy),
                            shared_mem=8*(tile + 2*nh - 1))
        return xp, f

    @staticmethod
    def _cplx(st):
        return np.int32(np.iscomplexobj(st.taps) and bool(np.any(np.imag(st.taps))))

    @staticmethod
    def _params(xp, sets, sps, base):
        """Each set's first instant (relative to the samples held) and step, in 32.32 fixed point"""
        A = np.array([((m0*sps + v - r*at)/(1 - r) + off - base) for m0, _, (v, r, at, off) in sets])
        S = np.array([sps/(1 - r) for _, _, (v, r, at, off) in sets])
        fix = lambda a: xp.asarray(np.round(a*4294967296.0).astype(np.int64))
        return fix(A), fix(S)

    def _energy(self, st, sets, sps):
        """Mean |y|^2 over each set (0 for an empty one)"""
        n = np.array([c for _, c, _ in sets], dtype=np.int64)
        if not n.sum():
            return [0.0]*len(sets)
        if not has_device_kernels(st.xp):
            return [float(np.mean(np.abs(self._reference(st, t))**2)) if len(t) else 0.0 for t in self._instants_of(sets, sps)]
        xp, f = self._device(st)
        A, S = self._params(xp, sets, sps, st.base)
        sums = xp.empty(len(sets), dtype=np.float32)
        f["mf_energy"]((len(sets),), (self.THREADS,), (st.dy, np.int64(len(st.dy)), A, S, xp.asarray(n.astype(np.int32)), sums))
        sums = to_host(sums).astype(np.float64) # (a number a set comes back)
        return [float(v/c) if c else 0.0 for v, c in zip(sums, n)]

    def _values(self, st, sets, sps):
        """y at every instant of the sets, one set after another (complex64, on the host)"""
        n = np.array([c for _, c, _ in sets], dtype=np.int64)
        if not n.sum():
            return np.zeros(0, np.complex64)
        if not has_device_kernels(st.xp):
            return self._reference(st, np.concatenate(self._instants_of(sets, sps))).astype(np.complex64)
        xp, f = self._device(st)
        A, S = self._params(xp, sets, sps, st.base)
        start = np.concatenate(([0], np.cumsum(n)))
        out = xp.empty(int(start[-1]), dtype=np.complex64)
        T = self.THREADS
        f["mf_values"](((int(start[-1]) + T - 1)//T,), (T,), (st.dy, np.int64(len(st.dy)), A, S, xp.asarray(start),
                                                            np.int32(len(sets)), out))
        return to_host(out)

    # ---- timing

    @staticmethod
    def _time(tr, m, sps):
        """Instant of symbol m by the tracked line (as OerderMeyr): t = theta(t) + m*sps"""
        return (m*sps + tr.value - tr.rate*tr.at)/(1 - tr.rate)

    @staticmethod
    def _line(tr, off=0.0):
        return (float(tr.value), float(tr.rate), float(tr.at), float(off))

    def _window(self, tr, a, b, sps, off=0.0):
        """The set of the symbols a line puts in [a, b - 4) (what a measurement of the window reads, whatever the cut)"""
        lo = int(np.ceil((a*(1 - tr.rate) - tr.value + tr.rate*tr.at)/sps))
        hi = int(np.ceil(((b - 4)*(1 - tr.rate) - tr.value + tr.rate*tr.at)/sps))
        return (lo, max(hi - lo, 0), self._line(tr, off))

    def _acquire(self, st, a, b, sps):
        """(phase, contrast) of a window from `grid` phases: the energy's maximum, and how far it stands above its minimum"""
        lo, hi = int(np.ceil(a/sps)), int(np.ceil((b - 4 - sps)/sps))
        if hi - lo < 16:
            return None, 0.0
        E = np.array(self._energy(st, [(lo, hi - lo, (0.0, 0.0, 0.0, g*sps/self.grid)) for g in range(self.grid)], sps))
        g, d = peak(E, circular=True, log=False)
        contrast = float((E.max() - E.min())/max(E.mean(), 1e-30))
        return ((g + d)*sps/self.grid) % sps, contrast

    def step(self, x, st, meta):
        sps = float(self.sps or meta.sps)
        delta = self.delta if self.delta is not None else sps/4
        nh = len(st.taps)
        W = max(int(round(self.window*meta.Fs)), int(np.ceil(self.min_symbols*sps)))
        wins = [(a0, b) for a0, b, _ in track.windows(st, x, meta, W)]
        st.y = st.dy = None # (what was made of the samples held: this call's)
        end = st.base + len(st.buf)
        out = [] # the output symbols, as sets
        tr = st.tr
        i = 0
        while i < len(wins):
            a0, b = wins[i]
            if not tr.started or tr.missed >= self.relock:
                tau, st.contrast = self._acquire(st, a0, b, sps)
                if tau is not None and st.contrast >= self.gate:
                    c = (a0 + b)/2
                    if tr.started: # (again, after misses: on the old line's nearest symbol, so no symbol is given twice)
                        tau += sps*np.round((track.at(tr, c) - tau)/sps)
                    tr.started = False
                    track.update(tr, tau, c)
                else:
                    track.miss(tr, (a0 + b)/2)
                self._out(st, out, wins[i], end, sps, meta)
                i += 1
                continue
            # A run of this group's windows, measured at the instants predicted at the group's start (snapshot)
            g = (a0//W)//self.group if tr.windows >= self.group else None # (one at a time until the loop has settled)
            if g is None or st.gidx != g:
                st.snap, st.gidx = SimpleNamespace(**vars(tr)), g
            run = [i]
            while g is not None and run[-1] + 1 < len(wins) and wins[run[-1] + 1][0]//W//self.group == g:
                run.append(run[-1] + 1)
            sets = [self._window(st.snap, *wins[j], sps, off) for j in run for off in (-delta, 0.0, delta)]
            E = self._energy(st, sets, sps)
            for n, j in enumerate(run):
                a0, b = wins[j]
                em, e0, ep = E[3*n:3*n + 3]
                den = em - 2*e0 + ep
                c = (a0 + b)/2
                if sets[3*n][1] >= 16 and den < 0: # A maximum: its vertex, from where the snapshot put the symbols
                    d = float(np.clip(delta*(em - ep)/(2*den), -sps/2, sps/2))
                    bw = self.bw if tr.windows >= self.ACQUIRE else max(self.bw, 0.6)
                    track.update(tr, track.at(st.snap, c) + d, c, sps, bw)
                else:
                    track.miss(tr, c)
                self._out(st, out, wins[j], end, sps, meta)
                if tr.missed >= self.relock: # (the rest are acquired again)
                    run = run[:n + 1]
                    break
            i = run[-1] + 1
        y = self._values(st, out, sps)
        keep = st.k*st.W if tr.started else (st.k - 1)*st.W # (until it starts, the window before too: a preamble there)
        if st.m is not None and tr.started:
            keep = min(keep, int(np.floor(self._time(tr, st.m, sps))))
        track.keep(st, keep - R - nh)
        meta.stream.est[self.name] = dict(phase=tr.value % sps if tr.started else None, ppm=-tr.rate*1e6 if tr.started else None,
                                          contrast=st.contrast, windows=tr.windows, missed=tr.missed, latency=W/meta.Fs)
        meta.delay += (nh - 1)/2/meta.Fs
        meta.op.Fs = meta.Fs/sps
        meta.op.sps = 1
        return y

    def _out(self, st, out, win, end, sps, meta):
        """The set of the symbols of a window just measured (as OerderMeyr: from the line after its update)"""
        tr = st.tr
        if not tr.started:
            return
        a0, b = win
        start = a0 if st.m is not None else st.base
        lim = min(b, end - (3 if meta.flush else R + 1))
        if st.m is None:
            st.m = int(np.ceil((start + 1 - track.at(tr, start))/sps))
        hi = max(int(np.ceil((lim*(1 - tr.rate) - tr.value + tr.rate*tr.at)/sps)), st.m)
        lo = st.m
        while lo < hi and self._time(tr, lo, sps) - 1 < st.base: # (only at the start: no symbol before the samples)
            lo += 1
        if lo < hi:
            out.append((lo, hi - lo, self._line(tr)))
            st.m = hi
