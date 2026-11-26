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
    for name in code.names("conv", examples=False):
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

def _mask(pat, T):
    """Which of the (T, 2) coded bits of T input bits a pattern keeps, in the order they are sent (time first)"""
    return np.tile(pat, (1, T//pat.shape[1])).T

def mismatches(K, gen, rows):
    """mismatch() of each (pattern, hard bits) in rows under the mother code (K, gen) with that pattern, decoded together:
    every pattern of a mother decodes on the mother's trellis with erasures where it punctures (what a punctured decoder
    does), so the rows go through in one batch per length. Returns (differing bits, bits compared) per row"""
    mother = _code(K, gen, None)
    k, n = np.zeros(len(rows), dtype=np.int64), np.zeros(len(rows), dtype=np.int64)
    groups = {} # {(length, period, bits a period): {pattern: [row]}}
    for i, (pat, h) in enumerate(rows):
        groups.setdefault((len(h), pat.shape[1], int(pat.sum())), {}).setdefault(pat.tobytes(), []).append(i)
    for (L, P, m), by_pat in groups.items():
        T = L//m*P # (input bits the whole periods carry)
        if T == 0:
            continue
        order = [i for idx in by_pat.values() for i in idx] # (the batch: each pattern's rows together)
        llr = np.zeros((len(order), T, 2))
        parts, a0 = [], 0
        for idx in by_pat.values():
            pat = rows[idx[0]][0]
            h = np.array([np.asarray(rows[i][1][:T//P*m], dtype=np.uint8) for i in idx])
            llr[a0:a0 + len(idx), _mask(pat, T)] = 1.0 - 2.0*h # (punctured bits stay erasures)
            parts.append((idx, pat, h, a0))
            a0 += len(idx)
        u = mother.decoder.decode_batch(llr.reshape(len(order), -1), soft=True)
        e = mother.encoder.encode_batch(u)
        e = e[:, :e.shape[1]//2*2].reshape(len(order), -1, 2)
        Te = e.shape[1]//P*P
        for idx, pat, h, a0 in parts: # Re-encoded, punctured as sent, compared
            sent = e[a0:a0 + len(idx), :Te][:, _mask(pat, Te)]
            nn = min(sent.shape[1], h.shape[1])
            k[idx] = np.count_nonzero(sent[:, :nn] != h[:, :nn].astype(bool), axis=1)
            n[idx] = nn
    return k, n

def _baselines(K, gen, keys):
    """What random bits get under each (pattern, length) in keys: its mismatch rate (at least 1e-4). The bits are seeded by
    the mother, the pattern and the length, so a baseline is the same whatever else is tried (and all decode together)"""
    rows = []
    for pat, L in keys:
        seed = [K, *np.asarray(gen, dtype=np.uint8).reshape(-1).tolist(), *pat.astype(int).reshape(-1).tolist(), pat.shape[1], L]
        rows.append((pat, np.random.default_rng(seed).integers(0, 2, L).astype(np.uint8)))
    k, n = mismatches(K, gen, rows)
    return {key: max(kk/max(nn, 1), 1e-4) for key, kk, nn in zip(((pat.tobytes(), pat.shape, L) for pat, L in keys), k, n)}

def search(streams, periods=range(2, 8), steps=400, confirm=4, codes=None):
    """Find a rate 1/2 mother code with a puncture pattern in hard bit streams ({tag: bits}). Returns (hypotheses, trials)

    Each (mother, pattern) is screened at every offset on `steps` input bits of each stream; what gets clearly fewer
    mismatches than on random bits is tried again on `confirm` times as many (at high rates a wrong pattern reproduces
    random bits fairly well too, so the short screen alone says little), and is a hypothesis if it still stands out.
    A mother's patterns, streams and offsets decode together (`mismatches`: a batch per length, not a call each), first
    the screens, then the confirmations of what passed."""
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
    key = lambda pat, L: (pat.tobytes(), pat.shape, L)
    for (K, gen), mname in codes.items():
        gen_list = [[int(x) for x in r] for r in gen]
        odd = all(int(np.sum(g)) % 2 == 1 for g in gen) # All-ones in, all-ones out: an inverted codeword is one
        jobs = [] # (P, pattern, tag, start, offset), in the order the hypotheses come
        for P in periods:
            per = steps//P
            for pat in patterns(P):
                m = int(pat.sum())
                for tag, bits in streams.items():
                    if odd and tag in twin:
                        continue
                    start = len(bits)//10
                    offs = [off for off in range(m) if len(bits[start + off:start + off + per*m]) == per*m]
                    trials += len(offs)
                    jobs += [(P, pat, tag, start, off) for off in offs]
        if not jobs:
            continue
        seg = lambda j, n: np.asarray(streams[j[2]][j[3] + j[4]:j[3] + j[4] + n*int(j[1].sum())], dtype=np.uint8)
        ks, ns = mismatches(K, gen, [(j[1], seg(j, steps//j[0])) for j in jobs])
        base = _baselines(K, gen, list({key(j[1], (steps//j[0])*int(j[1].sum())): (j[1], (steps//j[0])*int(j[1].sum()))
                                        for j in jobs}.values()))
        passed = [j for j, k, n in zip(jobs, ks, ns) if k/n < 0.5*base[key(j[1], (steps//j[0])*int(j[1].sum()))]]
        # Confirmed on a longer stretch
        long = [(j, seg(j, confirm*steps//j[0])) for j in passed]
        long = [(j, s_) for j, s_ in long if len(s_) == (confirm*steps//j[0])*int(j[1].sum())]
        if not long:
            continue
        kc, nc = mismatches(K, gen, [(j[1], s_) for j, s_ in long])
        bc = _baselines(K, gen, list({key(j[1], len(s_)): (j[1], len(s_)) for j, s_ in long}.values()))
        for (j, s_), k, n in zip(long, kc, nc):
            P, pat, tag, start, off = j
            m = int(pat.sum())
            b = bc[key(pat, len(s_))]
            z = float(-deviance_z(int(k), int(n), b))
            if z > 3.0 and k/n < 0.3*b:
                hyps.append(Hypothesis("conv-punctured", f"K={K} {mname} rate {P}/{m} {pat.astype(int).tolist()}", z,
                    {"K": K, "generator": gen_list, "puncture": pat.astype(int).tolist(),
                     "offset": (start + off) % m, "mismatch": k/n, "baseline": b, "stream": tag},
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
