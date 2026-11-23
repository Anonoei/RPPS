"""Frame structure on bits, blind: sync markers, frame length by autocorrelation, HDLC flags and the markers of a
framed code"""
import numpy as np

from ...math.prob import deviance_z, z_critical
from ..result import Hypothesis

def find_sync(bits, markers=(0x47, 0xB8), period=204):
    """Bit offset and byte phase at which `markers` repeat every `period` bytes. Returns (fraction, bit offset, phase)"""
    bits = np.asarray(bits, dtype=np.uint8)
    best = (0.0, 0, 0)
    for off in range(8):
        b = bits[off:]
        B = np.packbits(b[:len(b)//8*8])
        n = len(B)//period*period
        if n < 4*period:
            continue
        hit = np.isin(B[:n], markers).reshape(-1, period).mean(axis=0)
        p = int(np.argmax(hit))
        if hit[p] > best[0]:
            best = (float(hit[p]), off, p)
    return best

def frame_length(bits, min_n=16, top=5):
    """Candidate frame/block lengths from autocorrelation peaks (sync markers, repeated headers)

    Only sees fixed patterns that repeat every n bits; a code with no marker has none. Needs >= ~8 frames.
    Returns [(n, z)] strongest first, after removing multiples of a stronger shorter peak.
    """
    x = 1.0 - 2.0*np.asarray(bits, dtype=np.float64)
    N = len(x)
    x -= x.mean()
    F = np.fft.rfft(x, 2*N)
    ac = np.fft.irfft(np.abs(F)**2)[:N//4]/N/x.var()
    lags = np.arange(len(ac))
    z = np.abs(ac)*np.sqrt(N - lags)
    z[:min_n] = 0
    crit = z_critical(len(z))
    cand = []
    for i in np.argsort(-z)[:200]:
        if z[i] < crit:
            break
        cand.append((int(i), float(z[i])))
    cand.sort(key=lambda c: c[0])
    out = []
    for n, zz in cand:
        if not any(n % m == 0 for m, _ in out):
            out.append((n, zz))
    out.sort(key=lambda c: -c[1])
    return out[:top]

def scan_hdlc(streams, names=None, min_frames=3):
    """HDLC framing (config/frame): many flags, and frames whose FCS checks out

    Looks at flag counts first (a 0x7E pattern shows up at random once per 256 bits), then deframes the
    streams that have far more than that. Returns (hypotheses, trials).
    """
    from ... import link as encap
    names = names or encap.ls().get("frame", [])
    hyps, trials = [], 0
    for name in names:
        h = encap.load(f"frame/{name}").encoder
        if not isinstance(h, encap.hdlc): # (byte-stuffed framing: no bit flags to count)
            continue
        f = h.flag_bits
        for tag, bits in streams.items():
            for pol in (0, 1):
                b = np.asarray(bits, dtype=np.uint8) ^ pol
                work = b
                if h.nrzi:
                    work = np.concatenate(([0], (b[1:] == b[:-1]).astype(np.uint8)))
                win = np.lib.stride_tricks.sliding_window_view(work, len(f))
                nflag = int((win == f).all(axis=1).sum())
                trials += 1
                if nflag < 6:
                    continue
                frames, aborts = h.deframe(b[:200000])
                good = [p for p, ok in frames if ok]
                if len(good) >= min_frames:
                    z = float(deviance_z(len(good), max(len(frames), 1), 2.0**-(8*h.fcs.width//8)))
                    hyps.append(Hypothesis("hdlc", name, z, {"frames": len(good), "bad": len(frames)-len(good), "polarity": pol,
                        "stream": tag}, ("frame", name)))
    return hyps, trials

MARKER_PERIODS = 64  # Codewords enough to find a pattern of up to 16 markers (frame_markers: three of each, past the first half)

def frame_markers(bits, period, offset, column, unit=8, threshold=0.9, lengths=(1, 2, 4, 8, 16)):
    """The values that start every `period` units at `column` (units after bit `offset`): a pattern for `sync.Align`

    The shortest repeating pattern each of whose positions holds one value at least `threshold` of the time (DVB-S:
    0xB8 then 0x47 seven times), over the later half of the periods. Returns (pattern, worst fraction), or (None, 0.0)."""
    b = np.asarray(bits, dtype=np.uint8)[offset:]
    span = period*unit
    P = len(b)//span
    if P < 8:
        return None, 0.0
    vals = b[:P*span].reshape(P, period, unit)[:, column].astype(np.int64) @ (1 << np.arange(unit-1, -1, -1))
    if P >= 32: # The later half (the start of a window is where the sync before it was settling), but enough periods
        vals = vals[max(0, min(P//2, P - 3*max(lengths))):] # left for the longest pattern (a low code rate brings fewer)
    P = len(vals)
    for L in lengths:
        if P < 3*L: # At least three periods at every position of the pattern
            break
        pat, frac = [], 1.0
        for r in range(L):
            cnt = np.bincount(vals[r::L], minlength=1 << unit)
            pat.append(int(np.argmax(cnt)))
            frac = min(frac, cnt.max()/len(vals[r::L]))
        if frac >= threshold:
            return pat, float(frac)
    return None, 0.0
