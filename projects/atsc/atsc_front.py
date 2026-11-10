"""ATSC front end in one pass of C: 20 MS/s complex capture -> the channel centred, low-passed and resampled to the symbol rate (10.7622 MS/s, ONE complex sample per symbol)

    f = Front(cf, center_hz, step)           # step: input samples per symbol (20e6 / 10.7622e6 = 1.8582, trimmed to the clock error)
    z = f.process(block)                     # complex64, continuous across blocks (oscillator phase, filter history and the fractional read position are carried)

A 6 MHz channel centred on 0 is a complex signal 6 MHz wide, so one complex sample per symbol (10.76 MS/s) holds all of it with no aliasing, and the real T/2 signal the offline receiver (atsc_probe) builds at 21.5 MS/s
is not needed: a complex equalizer at one sample per symbol has the same two real degrees of freedom per symbol (atsc_rx.py). The work per input sample is a complex multiply (the oscillator) and per output
sample 2 x 32 multiply-adds of a windowed-sinc polyphase kernel (512 phases, cut-off 3.7 MHz, Kaiser): ~0.4 CPU-seconds per second of air, against ~6 for the numpy chain it replaces, with no gigabyte temporaries.
It is compiled with gcc on first use into ~/.cache/rpps/ (the only compiled helper left in projects/: RPPS has no fused mixer + polyphase resampler yet); `RPPS_NO_C=1`, or a missing compiler, uses a numpy version that is ~20x slower (enough for tests, not for real time).
Meant to be upstreamed into RPPS.
"""
import ctypes
import hashlib
import os
import shutil
import subprocess

import numpy as np

SYMBOL_RATE = 4.5e6*684/286
H = 16 # kernel half length: 32 taps
NPH = 512 # fractional phases (a timing grid of 1/1024 sample)
FC = 0.185 # kernel cut-off [cycles per input sample] = 3.7 MHz at 20 MS/s: flat over the 6 MHz channel (+-3 MHz), 60 dB down by 7.7 MHz (the first alias of the 10.76 MS/s output)

SOURCE = r"""
#include <math.h>
#include <string.h>
/* out[k] = x[k] * exp(j (ph0 + w k)); returns the phase after n samples */
double mix(const float *x, long n, double ph0, double w, float *out)
{
    double pr = cos(ph0), pi_ = sin(ph0), rr = cos(w), ri = sin(w);
    for (long k = 0; k < n; k++) {
        float xr = x[2*k], xi = x[2*k+1];
        out[2*k]   = (float)(xr*pr - xi*pi_);
        out[2*k+1] = (float)(xr*pi_ + xi*pr);
        double nr = pr*rr - pi_*ri, ni = pr*ri + pi_*rr; pr = nr; pi_ = ni;
        if ((k & 1023) == 1023) { double m = 1.0/sqrt(pr*pr + pi_*pi_); pr *= m; pi_ *= m; }
    }
    return fmod(ph0 + w*(double)n, 6.283185307179586);
}
/* polyphase windowed-sinc read of the complex buffer at positions pos0 + n step */
void resample(const float *buf, double pos0, double step, long nout, const float *tab, int H, int NPH, float *out)
{
    int T = 2*H;
    for (long n = 0; n < nout; n++) {
        double p = pos0 + n*step; long i0 = (long)p; double f = p - (double)i0;
        int ph = (int)(f*NPH + 0.5);
        const float *t = tab + (long)ph*T, *s = buf + 2*(i0 - H + 1);
        float ar = 0, ai = 0;
        for (int j = 0; j < T; j++) { ar += s[2*j]*t[j]; ai += s[2*j+1]*t[j]; }
        out[2*n] = ar; out[2*n+1] = ai;
    }
}
"""

_lib = None
_tried = False


def _load():
    global _lib, _tried
    if _tried:
        return _lib
    _tried = True
    if os.environ.get("RPPS_NO_C"):
        return None
    cc = shutil.which("gcc") or shutil.which("cc")
    if not cc:
        return None
    d = os.path.join(os.path.expanduser("~"), ".cache", "rpps"); os.makedirs(d, exist_ok=True)
    so = os.path.join(d, f"atsc_front_{hashlib.sha1(SOURCE.encode()).hexdigest()[:12]}.so")
    if not os.path.exists(so):
        src = so[:-3] + ".c"; tmp = so + f".{os.getpid()}"
        with open(src, "w") as f:
            f.write(SOURCE)
        if subprocess.run([cc, "-O3", "-march=native", "-ffast-math", "-shared", "-fPIC", "-o", tmp, src, "-lm"], capture_output=True).returncode:
            return None
        os.replace(tmp, so)
    try:
        lib = ctypes.CDLL(so)
    except OSError:
        return None
    lib.mix.restype = ctypes.c_double
    lib.mix.argtypes = [ctypes.c_void_p, ctypes.c_long, ctypes.c_double, ctypes.c_double, ctypes.c_void_p]
    lib.resample.restype = None
    lib.resample.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_double, ctypes.c_long, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    _lib = lib
    return lib


def kernel():
    """(NPH + 1, 2H) polyphase table: row p is the kernel for a read position p/NPH of the way past a sample; every row sums to 1 (unit gain at DC)"""
    f = np.arange(NPH + 1)/NPH
    t = np.arange(-H + 1, H + 1)
    d = f[:, None] - t[None, :] # distance from the read position to each tap's sample
    win = np.i0(7.0*np.sqrt(np.clip(1 - (d/H)**2, 0, 1)))/np.i0(7.0)
    g = 2*FC*np.sinc(2*FC*d)*win
    g /= g.sum(axis=1, keepdims=True)
    return np.ascontiguousarray(g, np.float32)


class Front:
    def __init__(self, cf, center_hz, step, fs=20e6):
        self.w = -2*np.pi*(center_hz - cf)/fs # the oscillator that moves the channel centre to 0
        self.step = float(step); self.ph = 0.0
        self.tail = np.zeros(0, np.complex64); self.pos = float(H) # read position in tail + next block
        self.tab = kernel(); self.lib = _load(); self.n_out = 0

    def process(self, x):
        x = np.ascontiguousarray(x, np.complex64)
        if self.lib is not None:
            mixed = np.empty(len(x), np.complex64)
            self.ph = self.lib.mix(x.ctypes.data, len(x), self.ph, self.w, mixed.ctypes.data)
        else:
            mixed = (x*np.exp(1j*(self.ph + self.w*np.arange(len(x))))).astype(np.complex64); self.ph = (self.ph + self.w*len(x)) % (2*np.pi)
        buf = np.concatenate((self.tail, mixed)) if len(self.tail) else mixed
        nout = int(np.floor((len(buf) - 1 - H - self.pos)/self.step)) + 1
        if nout <= 0:
            self.tail = buf
            return np.zeros(0, np.complex64)
        out = np.empty(nout, np.complex64)
        if self.lib is not None:
            self.lib.resample(buf.ctypes.data, self.pos, self.step, nout, self.tab.ctypes.data, H, NPH, out.ctypes.data)
        else:
            p = self.pos + self.step*np.arange(nout); i0 = p.astype(np.int64); ph = np.floor((p - i0)*NPH + 0.5).astype(np.int64)
            idx = i0[:, None] + np.arange(-H + 1, H + 1)[None, :]
            out[:] = np.einsum("nj,nj->n", buf[idx], self.tab[ph])
        nxt = self.pos + self.step*nout
        keep = max(0, int(nxt) - H + 1)
        self.tail = buf[keep:]; self.pos = nxt - keep; self.n_out += nout
        return out
