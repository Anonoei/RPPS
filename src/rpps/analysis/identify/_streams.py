"""Bit streams from symbols, and symbols that repeat (rpps.analysis.identify)"""
import numpy as np

from ...symbol.mapping import hard_bits

# --- Bit streams from symbols -------------------------------------------------------------------------

def _constellation(c):
    from ... import symbol
    return symbol.load_or_generate(c) if isinstance(c, str) else c


def diff_alphabets(constellation):
    """Phase-change alphabets a differential receiver of `constellation` would demap against: {name: Constellation}

    The PSK definitions with as many points (config/constellation/psk: QPSK is the pi/4-DQPSK alphabet, qpsk_axes 802.11b's;
    labelling variants such as the DVB-S2 maps are skipped), and the constellation itself when 1 is one of its points
    (DBPSK is BPSK)."""
    from ... import symbol
    c = _constellation(constellation)
    out = {}
    for stem in symbol.ls().get("psk", []):
        if stem.startswith("dvbs2_"):
            continue
        d = symbol.load(f"psk/{stem}")
        if d.order == c.order and np.allclose(np.abs(d.points), 1, atol=1e-6):
            out[stem] = d
    if np.min(np.abs(c.points - 1)) < 1e-6:
        out[c.name] = c
    return out


def bit_streams(symbols, constellation, differential=True, keys=None):
    """Hard bits of symbols under every map a receiver that doesn't know the carrier phase or the IQ sense has to try

    ("abs", i): the constellation's map i (its `maps`: every rotation, with and without spectral inversion)
    ("diff", name, i): phase changes y[n]/y[n-1] demapped with map i of a differential alphabet (`diff_alphabets`), only its
        unrotated maps (a phase change doesn't see the carrier phase, only the IQ sense)
    keys: only these. Returns {key: bits}
    """
    c = _constellation(constellation)
    y = np.asarray(symbols, dtype=np.complex128)
    out = {}
    for i, m in enumerate(c.maps):
        if keys is None or ("abs", i) in keys:
            out[("abs", i)] = hard_bits(y, c.points, m.arr)
    if differential and len(y) > 1:
        r = y[1:]/np.where(np.abs(y[:-1]) < 1e-12, 1.0, y[:-1])
        r = r/np.maximum(np.abs(r), 1e-12)
        for name, d in diff_alphabets(c).items():
            for i, m in enumerate(d.maps):
                if abs(np.angle(np.exp(1j*m.rotation))) > 1e-6:
                    continue
                if keys is None or ("diff", name, i) in keys:
                    out[("diff", name, i)] = hard_bits(r, d.points, m.arr)
    return out

# --- One answer: which code a symbol stream carries ------------------------------------------------------

CONV_SYMBOLS = 2**15 # Symbols the convolutional and block scans look at (the inner layers use all there are)
REPEAT_AGREE = 0.5 # |sum y[n] y*[n+P]| / sum |y|^2 above this: the symbols repeat with period P


def repeat_period(symbols, agree=REPEAT_AGREE, periods=3):
    """The period (symbols) the symbol stream repeats with, or None: fill, a test pattern or one frame sent again and again

    The symbols against themselves P later, |sum y[n] conj(y[n+P])| over the energy: about 1/sqrt(N) on data, the share
    of signal in signal and noise on a repeating stream, whatever the constellation and through a carrier offset (it
    turns every product alike). The shortest P whose agreement is over `agree`, seen at least `periods` times over.
    One period is all the evidence such a stream holds: a code test counted over every copy finds structure that is
    only the copying (rpps.analysis.identify runs on one period)"""
    y = np.asarray(symbols, dtype=np.complex128)
    n = len(y)
    if n < 4*periods:
        return None
    F = np.fft.fft(y - y.mean(), 2*n)
    ac = np.abs(np.fft.ifft(np.abs(F)**2)[:n//2])
    ac = ac/np.arange(n, n - len(ac), -1)/(np.mean(np.abs(y - y.mean())**2) + 1e-30) # per overlapping pair
    # Every multiple too: neighbours alike (symbols read at too high a rate, a narrow filter) agree at one lag only
    for P in np.nonzero(ac[1:] > agree)[0] + 1:
        if (periods - 1)*P >= len(ac):
            break
        if all(ac[k*P] > agree for k in range(2, periods)):
            return int(P)
    return None
