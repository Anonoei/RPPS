"""Blind self-synchronising (feedthrough) scrambler search

A feedthrough scrambler sends y[n] = d[n] ^ y[n-a] ^ y[n-b] ^ ...: descrambling is d[n] = y[n] ^ y[n-a] ^ y[n-b] ^ ..., with
no seed to find (the register fills from the received bits). Descrambling white bits with a wrong polynomial gives white
bits; the right one gives the data back, which almost always has structure: idle ones or zeros, HDLC flags, fixed headers,
frames. So every polynomial up to a degree is tried and the one whose output is furthest from white wins.

"White" here means looking random: about half ones, no byte pattern more common than another, no autocorrelation. Bits
that come out all ones (an idle line) or full of one pattern are the opposite, and the strongest evidence there is.

A multiple of the right polynomial also leaves structure (it is the right one followed by another descrambler, which keeps
a constant constant), so among the polynomials that stand out the one of lowest degree, then fewest taps, is taken.
Output that stays white under every polynomial was not scrambled this way, or what is under it is itself random
(compressed or encrypted data, or another layer such as an additive scrambler or a code).
"""
import itertools

import numpy as np

from ..math.prob import z_critical
from .fec import Hypothesis, Guess

def descramble(bits, taps):
    """d[n] = y[n] ^ y[n-t] for every tap t (the first max(taps) bits are not fully descrambled)"""
    y = np.asarray(bits, dtype=np.uint8)
    d = y.copy()
    for t in taps:
        d[t:] ^= y[:-t]
    return d

def structure(bits, lags=4096):
    """How far bits are from white, as z-scores: {"bias", "bytes", "autocorr", "score" (the largest)}

    bias: ones vs zeros. bytes: chi-square of the byte values (the most uneven of the 8 bit alignments). autocorr: the
    strongest autocorrelation up to `lags` (frames, repeated headers; 0 when lags is 0)"""
    b = np.asarray(bits, dtype=np.uint8)
    N = len(b)
    out = {"bias": float(abs(2*b.mean() - 1)*np.sqrt(N)) if N else 0.0}
    chi = 0.0
    for off in range(8):
        B = np.packbits(b[off:][:(N - off)//8*8])
        if len(B) < 512:
            continue
        c = np.bincount(B, minlength=256)
        e = len(B)/256
        chi = max(chi, (float(((c - e)**2).sum()/e) - 255)/np.sqrt(510))
    out["bytes"] = chi
    out["autocorr"] = 0.0
    if lags and N > 4*lags:
        x = 1.0 - 2.0*b
        x = x - x.mean()
        v = x.var()
        if v > 0:
            F = np.fft.rfft(x, 2*N)
            ac = np.fft.irfft(np.abs(F)**2)[1:lags + 1]/N/v
            out["autocorr"] = float(np.abs(ac).max()*np.sqrt(N))
    out["score"] = max(out.values())
    return out

def polynomials(max_degree=31, max_taps=2):
    """Tap sets to try: every 1..max_taps taps up to max_degree (1 + x^a is differential decoding / NRZI at a=1)"""
    out = []
    for k in range(1, max_taps + 1):
        out += [tuple(sorted(c, reverse=True)) for c in itertools.combinations(range(1, max_degree + 1), k)]
    return out

def search(bits, max_degree=31, max_taps=2, top=20):
    """Find the feedthrough scrambler on hard bits. Returns (hypotheses best first, trials, Guess or None)

    Every polynomial (and none) is scored on bias and byte statistics; the `top` best get the autocorrelation too. A
    polynomial stands out when its score is far above what the others typically get (robust z over all of them) and the
    structure is itself significant, both against the threshold for that many trials times the lags each score is the
    largest of. The Guess is the feedthrough definition of the
    lowest-degree one that stands out (see the module docstring), never decoded unless saved and loaded."""
    y = np.asarray(bits, dtype=np.uint8)
    cands = [()] + polynomials(max_degree, max_taps)
    quick = np.array([structure(descramble(y, t), lags=0)["score"] for t in cands])
    order = np.argsort(-quick)[:top]
    full = quick.copy()
    for i in order:
        full[i] = structure(descramble(y, cands[i]))["score"]
    med = float(np.median(quick))
    mad = float(np.median(np.abs(quick - med)))*1.4826 or 1.0
    crit = z_critical(len(cands)*4096) # Each score is itself the largest of many (byte alignments, autocorrelation lags)
    hyps = []
    for i in np.argsort(-full)[:top]:
        rz = (full[i] - med)/mad
        if rz > crit and full[i] > crit:
            t = cands[i]
            hyps.append(Hypothesis("scrambler", "none" if not t else "1 + " + " + ".join(f"x^{a}" for a in t), float(rz),
                                   {"taps": list(t), "score": float(full[i]), "median": med}))
    guess = None
    if hyps:
        best = hyps[0].z
        near = [h for h in hyps if h.z >= 0.5*best]
        pick = min(near, key=lambda h: (max(h.params["taps"], default=0), len(h.params["taps"]), -h.z))
        hyps.remove(pick)
        hyps.insert(0, pick)
        if pick.params["taps"]:
            taps = pick.params["taps"]
            guess = Guess("scramble", {"base": "scramble", "type": "feedthrough", "register": "fibonacci",
                                       "comment": f"found by rpps.analysis.scrambler.search: {pick.name}",
                                       "poly": taps, "seed": [1]*max(taps)},
                          {"z": pick.z, "score": pick.params["score"], "median": med})
    return hyps, len(cands), guess
