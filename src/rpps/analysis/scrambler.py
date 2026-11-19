"""Blind scrambler search: self-synchronising (feedthrough) scramblers, and additive ones (and bare LFSR sequences: PRBS)

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

An additive (synchronous) scrambler sends y[n] = d[n] ^ s[n], s an LFSR sequence. Wherever the data is constant for a
while (idle fill, padding, a fixed header) y is the LFSR sequence itself, or its complement, and a stretch of it has
linear complexity L (its register length) where random bits have half their length (`additive_search`, by
Berlekamp-Massey over sliding windows). A bare PRBS (a test pattern: ITU-T O.150/O.151) is the same everywhere.
"""
import itertools

import numpy as np

from ..math.prob import z_critical
from ..math.gf2 import pdivmod, berlekamp_massey, lfsr_taps
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


# --- Additive scramblers and PRBS: LFSR sequences, by linear complexity ------------------------------------------------

# Recurrences s[n] = s[n-a] ^ s[n-b] ^ ... known by name, by their taps (a polynomial and its reciprocal are one sequence
# read backwards, so both are looked up)
KNOWN_LFSR = {
    (7, 6): "PRBS7 (ITU-T O.150, 1+x^6+x^7)", (9, 5): "PRBS9 (ITU-T O.150/O.153, 1+x^5+x^9)",
    (11, 9): "PRBS11 (ITU-T O.150/O.152, 1+x^9+x^11)", (15, 14): "PRBS15 / DVB energy dispersal (1+x^14+x^15)",
    (20, 3): "PRBS20 / V.35 polynomial (1+x^3+x^20)", (20, 17): "PRBS20 (ITU-T O.150, 1+x^17+x^20)",
    (23, 18): "PRBS23 (ITU-T O.150/O.151, 1+x^18+x^23)", (23, 5): "V.32 call / PRBS23 reversed (1+x^5+x^23)",
    (29, 27): "PRBS29 (1+x^27+x^29)", (31, 28): "PRBS31 (ITU-T O.150, 1+x^28+x^31)",
    (8, 7, 5, 3): "CCSDS randomizer (1+x^3+x^5+x^7+x^8)", (17, 14): "V.22 / 1+x^14+x^17", (7, 4): "IEEE 802.11 / 1+x^4+x^7",
    (16, 15, 13, 4): "PRBS16 (1+x^4+x^13+x^15+x^16)", (9, 4): "1+x^4+x^9 (Bluetooth/BLE whitening, ...)",
}

def _poly(taps):
    """1 + sum x^t as an int (bit i: the coefficient of x^i)"""
    v = 1
    for t in taps:
        v |= 1 << int(t)
    return v

def _scrambler_polys():
    """{taps: name} of the feedthrough scramblers configured (config/scramble/feedthrough), and the known LFSRs"""
    from .. import code
    out = {}
    for name in code.scram.ls().get("feedthrough", []):
        try:
            from .fec import cached_scrambler
            t = tuple(sorted(cached_scrambler(f"feedthrough/{name}")["poly"], reverse=True))
            out.setdefault(t, f"feedthrough/{name} ({lfsr_name(t)})")
        except Exception: # A definition without a plain tap list isn't one to factor by
            continue
    for t in KNOWN_LFSR:
        out.setdefault(t, lfsr_name(t))
    return out

def explain(taps, max_period=4096):
    """Factor an LFSR recurrence into known scramblers and what is left: (scramblers [(taps, name, times)], the period
    the data under them repeats with (the least P whose x^P + 1 the rest divides; None if none up to max_period), rest)

    A feedthrough scrambler over data that repeats every P bits gives a sequence whose minimal polynomial is the
    scrambler's times a divisor of x^P + 1 (the data's own recurrence); two scramblers in cascade multiply."""
    rest = _poly(taps)
    found = []
    for t, name in sorted(_scrambler_polys().items(), key=lambda kv: -max(kv[0])):
        g, k = _poly(t), 0
        while rest.bit_length() > g.bit_length() - 1 and rest != 1:
            q, r = pdivmod(rest, g)
            if r:
                break
            rest, k = q, k + 1
        if k:
            found.append((list(t), name, k))
    period = None
    if rest == 1:
        period = 1
    else:
        x_p = 1 # x^P mod rest, P counting up: rest divides x^P + 1 where that is 1
        for P in range(1, max_period + 1):
            x_p = pdivmod(x_p << 1, rest)[1]
            if x_p == 1:
                period = P
                break
    return found, period, rest

def lfsr_name(taps):
    """A known name for these taps (or their reciprocal), else the polynomial written out"""
    t = tuple(sorted(taps, reverse=True))
    if not t:
        return "constant"
    L = t[0]
    rec = tuple(sorted([L - a for a in t[1:]] + [L], reverse=True))
    name = KNOWN_LFSR.get(t) or KNOWN_LFSR.get(rec)
    return name or "1 + " + " + ".join(f"x^{a}" for a in sorted(t))

def recurrence_holds(bits, taps):
    """Per bit n >= max(taps): whether s[n] = XOR s[n-t] holds (1) there"""
    s = np.asarray(bits, dtype=np.uint8)
    L = max(taps)
    acc = s[L:].copy()
    for t in taps:
        acc ^= s[L - t:len(s) - t]
    return (acc == 0).astype(np.uint8)

def additive_search(bits, window=512, max_len=None, step=None):
    """Find an LFSR sequence in hard bits: a bare PRBS, or an additive scrambler showing through constant data. Returns
    (hypotheses best first, windows tried, Guess or None)

    Every `step` bits a window of `window` bits is put through Berlekamp-Massey; one whose linear complexity is at most
    `max_len` (random bits: window/2, within a few) is a stretch of some LFSR's output. The polynomial most windows agree
    on is the scrambler's; how much of the stream its recurrence predicts tells a PRBS (all of it) from a scrambler over
    data (the constant stretches) and gives where those stretches are (a frame period, when they repeat)."""
    b = np.asarray(bits, dtype=np.uint8)
    step = step or window//2
    max_len = window//2 - 32 if max_len is None else max_len # Random windows come in within a few bits of window/2
    hits, tried = {}, 0
    for start in range(0, len(b) - window + 1, step):
        tried += 1
        C, L = berlekamp_massey(b[start:start + window])
        taps = lfsr_taps(C, L)
        if 0 < L <= max_len and 2*L + 64 <= window and taps:
            hits.setdefault(taps, []).append(start)
    if not hits:
        return [], tried, None
    hyps = []
    for taps, starts in hits.items():
        holds = recurrence_holds(b, taps)
        frac = float(holds.mean())
        # Each window is overwhelming evidence alone (random bits fall that short of window/2 about 2^-(window - 2L) of
        # the time), so the score is how many windows agree
        hyps.append(Hypothesis("scrambler-additive", lfsr_name(taps), float(len(starts)),
                               {"taps": list(taps), "length": max(taps), "windows": len(starts), "of": tried,
                                "holds": frac, "first": starts[0], "prbs": frac > 0.95}))
    hyps.sort(key=lambda h: (-h.params["windows"], h.params["length"]))
    best = hyps[0]
    kind = "a bare PRBS (test pattern)" if best.params["prbs"] else "an additive scrambler showing through constant data"
    scr, period, _ = explain(best.params["taps"])
    if scr and best.name not in KNOWN_LFSR.values():
        # Not a PRBS of its own: known feedthrough scramblers over data that repeats
        names = " then ".join(f"{name} x{k}" if k > 1 else name for _, name, k in scr)
        kind = (f"{names} over data repeating every {period} bits" if period and period > 1 else
                f"{names} over constant data" if period == 1 else f"{names} over data with recurrence of its own")
        best.params["scramblers"] = [{"taps": t, "name": n, "times": k} for t, n, k in scr]
        best.params["data_period"] = period
        if period and period <= 4096: # The data itself: descrambled, the commonest word of `period` bits
            d = b
            for t, _, k in scr:
                for _ in range(k):
                    d = descramble(d, t)
            d = d[64*max(max(t) for t, _, _ in scr):]
            words = d[:len(d)//period*period].reshape(-1, period)
            vals, counts = np.unique(words, axis=0, return_counts=True)
            w = vals[np.argmax(counts)]
            best.params["pattern"] = "".join(map(str, w))
            best.params["pattern_holds"] = float(counts.max()/len(words))
            kind += f": {best.params['pattern']} ({best.params['pattern_holds']*100:.1f}% of words)"
    best.params["what"] = kind
    best.name = f"{best.name}: {kind}"
    guess = Guess("scramble", {"base": "scramble", "type": "additive", "comment": f"found by rpps.analysis.scrambler.additive_search: "
                               f"{best.name}, {kind}; the seed and where it restarts are not known yet",
                               "poly": best.params["taps"]}, {"windows": best.params["windows"], "holds": best.params["holds"]})
    return hyps, tried, guess
