"""Test signals and the small conversions around them"""
import numpy as np

import rpps as rp
from rpps.base.types import Rep
from rpps.base import Mod

def bits(rng, n):
    return rng.integers(0, 2, n).astype(bool)

def awgn(syms, esn0_db, rng):
    """Add complex white noise to unit-power symbols at Es/N0. Returns (noisy, noise variance E|n|^2)"""
    var = 10**(-esn0_db/10)
    n = (rng.normal(size=len(syms)) + 1j*rng.normal(size=len(syms)))*np.sqrt(var/2)
    return syms + n.astype(np.complex64), var

def shaped(rng, points, n_sym, sps, beta=0.35, snr_db=None, esn0_db=None, cfo=0.0, tau=0.0, span=12, sym=None):
    """RRC-shaped symbols at any (fractional) samples per symbol: x(t) = sum s_k p(t - k - tau), t in symbols, sampled
    every 1/sps symbols; cfo in cycles a sample; noise at snr_db over the sample bandwidth or esn0_db per symbol.
    Returns (samples complex64, symbols sent)"""
    from rpps.signal.filters.shaping import rrc_pulse
    pts = np.asarray(points, dtype=np.complex128)
    pts = pts/np.sqrt(np.mean(np.abs(pts)**2))
    s = pts[rng.integers(0, len(pts), n_sym)] if sym is None else np.asarray(sym)
    t = np.arange(int(n_sym*sps))/sps - tau
    k0 = np.floor(t).astype(int)
    x = np.zeros(len(t), complex)
    for d in range(-span, span + 1):
        k = k0 + d
        ok = (k >= 0) & (k < n_sym)
        x[ok] += s[k[ok]]*rrc_pulse(t[ok] - k[ok], beta)
    x = x/np.sqrt(sps) # unit symbol energy: per-sample power 1/sps
    if esn0_db is not None:
        snr_db = esn0_db - 10*np.log10(sps)
    if snr_db is not None:
        var = np.mean(np.abs(x)**2)*10**(-snr_db/10)
        x = x + (rng.standard_normal(len(x)) + 1j*rng.standard_normal(len(x)))*np.sqrt(var/2)
    x = x*np.exp(2j*np.pi*cfo*np.arange(len(x)))
    return x.astype(np.complex64), s

def cpfsk(rng, n_sym, sps, M=2, h=0.5, bt=None, snr_db=20.0, cfo=0.0, sym=None):
    """Continuous-phase M-FSK: tones h baud apart, rectangular or Gaussian (bt) frequency pulse, cfo in cycles a sample,
    noise at snr_db over the sample bandwidth. Returns (samples complex64, levels sent: -(M-1) .. M-1 in steps of 2)"""
    a = (2*rng.integers(0, M, n_sym) - (M - 1)) if sym is None else np.asarray(sym)
    n = int(n_sym*sps)
    t = np.arange(n)/sps
    f = a[np.minimum(t.astype(int), n_sym - 1)].astype(float)*h/2/sps # cycles a sample
    if bt:
        s = np.sqrt(np.log(2))/(2*np.pi*bt)*sps
        k = np.arange(-int(4*s), int(4*s) + 1)
        g = np.exp(-k**2/(2*s**2))
        f = np.convolve(f, g/g.sum(), mode="same")
    x = np.exp(2j*np.pi*(np.cumsum(f) + cfo*np.arange(n)))
    x = x + 10**(-snr_db/20)*(rng.standard_normal(n) + 1j*rng.standard_normal(n))/np.sqrt(2)
    return x.astype(np.complex64), a

def rrc_psk(n, sps, M=4, beta=0.35, tau=0.0, cfo=0.0, esn0_db=20.0, seed=0, sent=False):
    """n samples of M-PSK through an RRC at a fractional sps, the first symbol at tau, a carrier offset [cycles/sample]
    sent: also the symbols, as (instants, symbols)"""
    rng = np.random.default_rng(seed)
    k = np.arange(-20, int(n/sps) + 20)
    sym = np.exp(2j*np.pi*rng.integers(0, M, len(k))/M + 1j*np.pi/M*(M == 4))
    t = np.arange(n)
    x = np.zeros(n, complex)
    L = int(np.ceil(10*sps))
    for kk, s in zip(k, sym):
        c = tau + kk*sps
        j0, j1 = int(max(0, np.floor(c - L))), int(min(n, np.ceil(c + L)))
        if j1 > j0:
            x[j0:j1] += s*rp.signal.filters.rrc_pulse((t[j0:j1] - c)/sps, beta)
    x *= np.exp(2j*np.pi*cfo*t)
    n0 = np.mean(np.abs(x)**2)*sps/10**(esn0_db/10)
    x = x + np.sqrt(n0/2)*(rng.standard_normal(n) + 1j*rng.standard_normal(n))
    return (x, (tau + k*sps, sym)) if sent else x

def psd_noise(rng, N, Fs, psd):
    """Complex Gaussian samples whose power spectrum is psd(f) [linear, 1: the noise floor; f in Hz]"""
    f = np.fft.fftfreq(N, 1/Fs)
    g = (rng.standard_normal(N) + 1j*rng.standard_normal(N))/np.sqrt(2)
    return (np.fft.ifft(g*np.sqrt(psd(f)))*np.sqrt(N)).astype(np.complex64)

def rc(f, baud, beta=0.35):
    """A raised-cosine spectrum, 1 in its flat part"""
    a = np.abs(f)
    lo, hi = (1 - beta)*baud/2, (1 + beta)*baud/2
    return np.where(a <= lo, 1.0, np.where(a <= hi, 0.5*(1 + np.cos(np.pi/(beta*baud)*(a - lo))), 0.0))

def hard(d):
    """The bits of a block of bits or LLRs"""
    x = np.asarray(d.data)
    return x < 0 if x.dtype.kind == "f" else x.astype(bool)

def unhex(h, n):
    return np.unpackbits(np.frombuffer(bytes.fromhex(h), np.uint8))[:n]

def as_data_bits(b, rep=Rep.DATA, **kw):
    d = rp.Data(np.asarray(b, dtype=bool), **kw)
    return d.derive(np.asarray(b, dtype=bool), rp.Category.CODE, rp.dtype.BITS, rep=rep)

def as_llr(d, scale=3.0):
    """Sure LLRs with the hard decisions of d"""
    return d.derive((1 - 2*np.asarray(d.data).astype(np.float32))*scale, rp.Category.SYMBOL, rp.dtype.LLR, Mod, rep=Rep.CHANNEL)

def two_bands(rng, N, Fs, f1, f2, bw):
    f = np.fft.fftfreq(N, 1/Fs)
    x = 0
    for fc in (f1, f2):
        spec = np.where(np.abs(f - fc) < bw/2, rng.normal(size=N) + 1j*rng.normal(size=N), 0)
        x = x + np.fft.ifft(spec)*np.sqrt(N)*0.5
    return (x + (rng.normal(size=N) + 1j*rng.normal(size=N))*0.05).astype(np.complex64)
