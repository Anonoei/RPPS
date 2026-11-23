"""Blind puncture patterns: a known rate 1/2 convolutional code (a "mother code") sent through a puncture pattern that no
definition has

A pattern of period P keeps some of the 2P coded bits of P input bits. Rate P/(P+1) (2/3, 3/4, ... 7/8) keeps P+1: one input
bit's two outputs and one output of each of the others, so there are P * 2^(P-1) patterns, and a pattern turned by a column
is the same one started a column later, which the bit offset covers: 2^(P-1) to try per period, each at every offset. The
right one lets the Viterbi decoder reproduce what was received (only channel errors differ); others don't.

Mother codes are those in config/fec/conv (both generator orders: which of the two outputs goes first is not known).
"""
import functools
import itertools

import numpy as np

from ...result import Hypothesis, Guess, Report, pick
from ....math.prob import deviance_z
from ....code.coding import mismatch, mismatch_batch

@functools.lru_cache(maxsize=None)
def patterns(P):
    """Rate P/(P+1) puncture patterns (2, P), one per turn (built once per P; read only)"""
    seen, out = set(), []
    for both in range(P):
        for choice in itertools.product((0, 1), repeat=P - 1):
            pat = np.zeros((2, P), dtype=bool)
            pat[:, both] = True
            for col, which in zip([c for c in range(P) if c != both], choice):
                pat[which, col] = True
            key = min(tuple(map(tuple, np.roll(pat, s, axis=1))) for s in range(P))
            if key not in seen:
                seen.add(key)
                out.append(np.array(key, dtype=bool))
    return out

def mothers():
    """Rate 1/2 mother codes in config: {(K, generators): name}, both output orders"""
    from .... import code
    out = {}
    for name in code.names("conv"):
        e = code.load(f"conv/{name}").encoder
        if e.den != 2 or e.num != 1:
            continue
        g = np.asarray(e.generator, dtype=np.uint8)
        for gg in (g, g[::-1]):
            out.setdefault((int(e.con), tuple(map(tuple, gg))), name.split("/")[0] + ("" if gg is g else " (outputs swapped)"))
    return out

def _code(K, gen, pat):
    """The rate 1/2 mother code (K, generators) under puncture pattern pat"""
    from ....code import Conv
    return Conv(np.asarray(gen, dtype=np.uint8).reshape(2, K), puncture=pat)

def search(streams, periods=range(2, 8), steps=400, confirm=4, codes=None):
    """Find a rate 1/2 mother code with a puncture pattern in hard bit streams ({tag: bits}). Returns (hypotheses, trials)

    Each (mother, pattern) is screened at every offset on `steps` input bits of each stream; what gets clearly fewer
    mismatches than on random bits is tried again on `confirm` times as many (at high rates a wrong pattern reproduces
    random bits fairly well too, so the short screen alone says little), and is a hypothesis if it still stands out."""
    rng = np.random.default_rng(0)
    codes = codes or mothers()
    hyps, trials = [], 0
    # Streams that are another's bits inverted (a half turn of QPSK): for a mother whose all-ones input encodes to all
    # ones (generators of odd weight) the inverse of a codeword is a codeword, so the twin repeats its partner's result
    twin, seen = {}, {}
    for tag, bits in streams.items():
        b = np.asarray(bits[:4096], dtype=np.uint8)
        inv = (b ^ 1).tobytes()
        if inv in seen:
            twin[tag] = seen[inv]
        seen.setdefault(b.tobytes(), tag)
    for (K, gen), mname in codes.items():
        gen_list = [[int(x) for x in r] for r in gen]
        odd = all(int(np.sum(g)) % 2 == 1 for g in gen) # All-ones in, all-ones out: an inverted codeword is one
        for P in periods:
            for pat in patterns(P):
                code = _code(K, gen, pat)
                m = int(pat.sum())
                base = {}
                def baseline(per):
                    if per not in base:
                        k0, n0 = mismatch(code, rng.integers(0, 2, per*m).astype(np.uint8))
                        base[per] = max(k0/n0, 1e-4)
                    return base[per]
                for tag, bits in streams.items():
                    if odd and tag in twin:
                        continue
                    start = len(bits)//10
                    per = steps//P
                    offs = [off for off in range(m) if len(bits[start + off:start + off + per*m]) == per*m]
                    if not offs: # Screened together: every offset in one batch
                        continue
                    ks, n = mismatch_batch(code, np.array([np.asarray(bits[start + off:start + off + per*m], dtype=np.uint8) for off in offs]))
                    trials += len(offs)
                    b0 = baseline(per)
                    for off, k in zip(offs, ks):
                        if k/n >= 0.5*b0:
                            continue
                        per_c = confirm*steps//P # Confirmed on a longer stretch
                        seg = np.asarray(bits[start + off:start + off + per_c*m], dtype=np.uint8)
                        if len(seg) < per_c*m:
                            continue
                        kc, nc = mismatch(code, seg)
                        b = baseline(per_c)
                        z = float(-deviance_z(kc, nc, b))
                        if z > 3.0 and kc/nc < 0.3*b:
                            hyps.append(Hypothesis("conv-punctured", f"K={K} {mname} rate {P}/{m} {pat.astype(int).tolist()}", z,
                                {"K": K, "generator": gen_list, "puncture": pat.astype(int).tolist(),
                                 "offset": (start + off) % m, "mismatch": kc/nc, "baseline": b, "stream": tag},
                                ("conv-punctured", f"{mname}|{pat.astype(int).tolist()}")))
    return hyps, trials

def best(hyps, trials):
    """The hypothesis that stands out and fits at one offset only, with its Guess (a conv definition), or (None, None)"""
    rep = Report(hyps, trials, None)
    v = pick(rep.hyps, rep.z_crit, lambda h: h.params["mismatch"]/h.params["baseline"],
              lambda b, h: h.params["mismatch"] < 0.5*(b.params["mismatch"] + b.params["baseline"]))
    if v is None:
        return None, None
    p = v.params
    from ....code.serial import mother
    m = mother(p["generator"], p["K"])
    head = {"base": "conv", "extends": m} if m else {"base": "conv"}
    body = {"puncture": p["puncture"]} if m else {"constraint": p["K"], "generator": p["generator"], "puncture": p["puncture"]}
    return v, Guess("conv", {**head, "comment": f"found by rpps.analysis.code.conv.puncture: {v.name}", **body},
                    {"z": v.z, "offset": p["offset"], "mismatch": p["mismatch"]})
