"""Mother codes of convolutional codes, blind: the constraint length and generators of a rate 1/n feed-forward code
(for generators g_a, g_b, (c_a * g_b) xor (c_b * g_a) = 0 on codewords), a configured code's fit, and the definition
a found one is saved as (a child of the configured mother code with these generators, else the generators).
Its punctured rates: rpps.analysis.code.conv.puncture"""
import itertools

import numpy as np

from ....math.prob import deviance_z
from ....math.gf2 import pgcd
from ....code.coding import puncture_period, mismatch, mismatch_batch, cached
from ...result import Hypothesis, Guess

def _polys(K):
    """Generators of constraint length K: first and last tap set (otherwise it is a shorter code)"""
    return [np.array((1,) + m + (1,), dtype=np.uint8) for m in itertools.product((0, 1), repeat=K-2)]

def _shared(polys):
    """(P, P) bool: generator pairs with a common factor. Such a pair is a catastrophic code, and its relation holds on
    whatever the factor leaves alike: equal generators test only c_a == c_b (BPSK read as QPSK, or fill), not a code"""
    ints = [int("".join(map(str, g)), 2) for g in polys]
    return np.array([[pgcd(a, b) != 1 for b in ints] for a in ints])

def _convolve_all(c, polys, K):
    """c * g for every generator, kept at length len(c)-K+1 (the part with full history)"""
    n = len(c)
    out = np.zeros((len(polys), n-K+1), dtype=np.uint8)
    for k in range(K):
        shifted = c[K-1-k:n-k]
        for i, g in enumerate(polys):
            if g[k]:
                out[i] ^= shifted
    return out

def _sum_all(r, polys, K):
    """sum of r over each generator's taps (r * g as a real sum), at length len(r)-K+1: (P, N)"""
    n = len(r)
    out = np.zeros((len(polys), n-K+1), dtype=np.float64)
    for k in range(K):
        shifted = r[K-1-k:n-k]
        for i, g in enumerate(polys):
            if g[k]:
                out[i] += shifted
    return out

def _soft_z(s, logmag):
    """z of parity relations s (P, N) weighted by their reliability exp(logmag) (P, N): sum x / sqrt(sum x^2), x the
    relation's sign times the product of tanh(|L|/2) over its bits. Under no code each x is as likely + as -, so z is
    about standard normal whatever the LLRs' scale (a miscalibrated scale costs power, never false alarms)"""
    x = (1.0 - 2.0*s)*np.exp(logmag)
    return x.sum(axis=1)/np.sqrt(np.maximum((x*x).sum(axis=1), 1e-300))

def _fold_z(s, T):
    """Per row of s (R, L): how non-uniform the relation is over the phase of a period-T pattern (chi^2 as z)"""
    m = s.shape[1]//T*T
    if m < 4*T:
        return np.zeros(len(s))
    mp = s[:, :m].reshape(len(s), -1, T).mean(axis=1)
    npp = m//T
    chi = np.sum(4*(mp-0.5)**2*npp, axis=1)
    return (chi-T)/np.sqrt(2*T)

def _period(s, z_min=7.0):
    """Strongest periodic structure in a 0/1 relation sequence: (period, z) or None.

    A convolutional relation holds uniformly over time. A block code's holds with a pattern that repeats with the block.
    """
    x = s.astype(np.float64) - s.mean()
    spec = np.abs(np.fft.rfft(x))
    spec[:3] = 0
    k = int(np.argmax(spec))
    z = float(spec[k]/np.sqrt(np.mean(spec[3:]**2) + 1e-30))/1.5
    return (len(s)/k, z) if z > z_min else None

def scan_conv(bits, ks=range(3, 8), rates=(2, 3), periods=(), tag=None, llr=None):
    """Rate 1/n convolutional code scan on one bit stream. Returns (hypotheses, number of trials)

    Every generator pair's parity relation (c_a * g_b) ^ (c_b * g_a) against chance. llr: the bits' LLRs (log P0/P1,
    as long as bits): each relation bit weighted by the product of tanh(|L|/2) over the bits it combines (`_soft_z`),
    so noisy bits count for less (the hard count weighs them all alike: soft reaches lower SNR)"""
    hyps, trials = [], 0
    bits = np.asarray(bits, dtype=np.uint8)
    rel = None
    if llr is not None:
        rel = np.log(np.clip(np.tanh(np.abs(np.asarray(llr, dtype=np.float64)[:len(bits)])/2), 1e-12, 1.0))
    for n in rates:
        offsets = range(n) if n == 2 else (0,)
        for off in offsets:
            b = bits[off:]
            m = len(b)//n
            streams = [b[i:m*n:n] for i in range(n)]
            rs = [rel[off:][i:m*n:n] for i in range(n)] if rel is not None else None
            for K in ks:
                polys = _polys(K)
                shared = _shared(polys)
                conv = [_convolve_all(c, polys, K) for c in streams]
                sums = [_sum_all(r, polys, K) for r in rs] if rs is not None else None
                for a, bb in itertools.combinations(range(n), 2):
                    for ia, ga in enumerate(polys):
                        # relation (c_a * g_b) ^ (c_b * g_a) for every g_b at once
                        s = conv[a] ^ conv[bb][ia]
                        # polys[ib] is g_b: relation uses c_a*g_b with c_b*g_a: here conv[a][ib], conv[bb][ia]
                        if sums is not None:
                            z = np.where(shared[ia], 0.0, _soft_z(s, sums[a] + sums[bb][ia]))
                        else:
                            k = s.sum(axis=1, dtype=np.int64)
                            z = np.where(shared[ia], 0.0, deviance_z(k, s.shape[1]))
                        trials += len(polys)
                        j = int(np.argmax(np.abs(z)))
                        if abs(z[j]) > 3.0:
                            params = {"n": n, "K": K, "offset": off, "streams": (a, bb),
                                      "g": {a: tuple(int(x) for x in ga), bb: tuple(int(x) for x in polys[j])},
                                      "stream": tag}
                            per = _period(s[j]) if abs(z[j]) > 6 else None
                            if per is None:
                                hyps.append(Hypothesis("conv", f"rate 1/{n} K={K}", float(abs(z[j])), params))
                            else: # Holds only in places: some other linear code (block, LDPC) with short checks
                                params["period_pairs"] = per[0]
                                hyps.append(Hypothesis("linear-block", f"short checks, period {per[0]:.1f} pairs", float(abs(z[j])), params))
                        for T in periods:
                            zf = np.where(shared[ia], 0.0, _fold_z(s, T))
                            trials += len(polys)
                            j = int(np.argmax(zf))
                            if zf[j] > 3.0:
                                hyps.append(Hypothesis("conv+scrambler", f"rate 1/{n} K={K} T={T}", float(zf[j]),
                                    {"n": n, "K": K, "offset": off, "T": T, "streams": (a, bb),
                                     "g": {a: tuple(int(x) for x in ga), bb: tuple(int(x) for x in polys[j])},
                                     "stream": tag}))
    return hyps, trials

def soft_mismatch_batch(code, llr):
    """The soft re-encoding mismatch of each row of LLRs (W, L): decoded from the LLRs, re-encoded, the share of |LLR| on
    the bits the code disagrees with (a right code's disagreements fall on unreliable bits, a wrong one's anywhere), and
    the effective number of bits ((sum |L|)^2 / sum L^2) that share is over. Returns (shares (W,), n_eff (W,), bits)"""
    llr = np.asarray(llr, dtype=np.float64)
    dec, enc = code.decoder, code.encoder
    u = dec.decode_batch(llr, soft=True) if hasattr(dec, "decode_batch") else np.array([dec.decode(r, True) for r in llr])
    e = enc.encode_batch(u) if hasattr(enc, "encode_batch") else np.array([enc.encode(np.asarray(x).astype(bool)) for x in u])
    n = min(e.shape[1], llr.shape[1])
    w = np.abs(llr[:, :n])
    off = e[:, :n].astype(bool) != (llr[:, :n] < 0)
    tot = np.maximum(w.sum(axis=1), 1e-300)
    return (w*off).sum(axis=1)/tot, tot**2/np.maximum((w*w).sum(axis=1), 1e-300), n

def scan_defined_conv(streams, names=None, steps=2500, skip=0.1, segments=(0.1, 0.45, 0.8), decisive=20.0, llrs=None,
                      screen=5, cut=0.85):
    """Try every convolutional definition in config/fec/conv on every bit mapping and puncture alignment

    A correct hypothesis lets the decoder reproduce the received bits (the channel error rate, well below
    what a wrong one gets). Each of `segments` (fractions into the stream) is tried on its own: a symbol the timing slipped
    moves the puncture phase, so a segment across it fits two phases half well, and the others still fit one. A
    hypothesis says which segment (its offset holds there). Once one fits overwhelmingly (z > `decisive`, a tenth of a
    wrong code's mismatch), only definitions of the same rate are tried further (an equivalent code, the generators in
    the other order, also fits and the pick needs to see it; a code of another rate cannot). llrs: {tag: LLRs} of the
    streams that have them: a hypothesis the hard re-encoding leaves short is also scored softly (soft_mismatch_batch,
    against what random signs on the same reliabilities give), which holds a few dB further down. Every hypothesis is
    screened on a `screen`-th of its bits first; one not under `cut` x its baseline there (hard or soft) is not decoded
    in full. Returns (hypotheses, trials).
    """
    from .... import code as coding
    names = names or coding.names("conv")
    hyps, trials = [], 0
    decided, baselines = None, {} # The rate (bits in, bits out a puncture period) of an overwhelming fit
    for name in names:
        code = cached(f"conv/{name}")
        enc = code.encoder
        pat = enc.puncture if enc.puncture is not None else np.ones((enc.den, 1), dtype=bool)
        rate = (pat.shape[1], int(pat.sum()))
        if decided is not None and rate != decided:
            continue
        L = pat.shape[1]
        nb = int(pat.sum())
        per = max(steps//L, 1)
        # What a wrong hypothesis gets: measured once per code shape (puncture period, bits a period) and
        # shared, so equivalent codes (the generators the other way round, an output inverted) tie exactly and the pick
        # is the same whichever codes were tried before (a draw per code made the winner of a tie a matter of chance)
        shape = (L, nb)
        if shape not in baselines:
            rand = np.random.default_rng(L*1000 + nb).integers(0, 2, per*nb).astype(np.uint8)
            k0, n0 = mismatch(code, rand)
            baselines[shape] = max(k0/n0, 1e-4)
        base = baselines[shape]
        for tag, bits in streams.items():
            starts = sorted({int(len(bits)*f) for f in segments if int(len(bits)*f) + nb + per*nb <= len(bits)})
            if not starts and len(bits) >= int(len(bits)*skip) + nb + per*nb:
                starts = [int(len(bits)*skip)]
            jobs = [(si, start, off) for si, start in enumerate(starts) for off in range(nb)
                    if len(bits[start+off:start+off+per*nb]) == per*nb] # Every segment and alignment, decoded together
            if not jobs:
                continue
            trials += len(jobs)
            L_ = (llrs or {}).get(tag)
            if L_ is not None and len(L_) < len(bits):
                L_ = None
            bs = None
            if L_ is not None:
                key = (shape, "soft", tag)
                if key not in baselines: # Random signs on these reliabilities: what a wrong code's soft share is
                    _, st0, of0 = jobs[0]
                    r = np.abs(np.asarray(L_[st0+of0:st0+of0+per*nb], dtype=np.float64))[None]
                    r = r*np.where(np.random.default_rng(L*1000 + nb + 1).random(r.shape) < 0.5, -1.0, 1.0)
                    baselines[key] = max(float(soft_mismatch_batch(code, r)[0][0]), 1e-4)
                bs = baselines[key]
            # Screen every job on a fifth of its bits: a wrong one comes in near its baseline, a right one well under it
            # (hard, or soft when the hard decisions are too noisy); only those decoded in full
            ps = max(per//screen, 40)
            if ps < per:
                hs, n_s = mismatch_batch(code, np.array([np.asarray(bits[st+of:st+of+ps*nb], dtype=np.uint8) for _, st, of in jobs]))
                keep = hs/n_s < cut*base
                if L_ is not None:
                    ss, _, _ = soft_mismatch_batch(code, np.array([np.asarray(L_[st+of:st+of+ps*nb], dtype=np.float64)
                                                                   for _, st, of in jobs]))
                    keep |= ss < cut*bs
                jobs = [j for j, k in zip(jobs, keep) if k]
                if not jobs:
                    continue
            ks, n = mismatch_batch(code, np.array([np.asarray(bits[start+off:start+off+per*nb], dtype=np.uint8) for _, start, off in jobs]))
            soft = None
            if L_ is not None:
                rows = np.array([np.asarray(L_[start+off:start+off+per*nb], dtype=np.float64) for _, start, off in jobs])
                sh, ne, _ = soft_mismatch_batch(code, rows)
                soft = (sh, ne, bs)
            for j, ((si, start, off), k) in enumerate(zip(jobs, ks)):
                k = int(k)
                z = float(-deviance_z(k, n, base)) # Fewer mismatches than a wrong code gives
                ok = z > 3.0 and k/n < 0.5*base # And clearly fewer, not a marginal difference on a short code
                params = {"offset": (start + off) % nb, "mismatch": k/n, "baseline": base, "stream": tag, "segment": si}
                if not ok and soft is not None: # Soft: the share of |LLR| it disagrees on, against random signs'
                    sh, ne, bs = soft
                    zs = float(-deviance_z(sh[j]*ne[j], ne[j], bs))
                    if zs > 3.0 and sh[j] < 0.5*bs:
                        ok, z = True, zs
                        params.update({"soft_mismatch": float(sh[j]), "soft_baseline": bs})
                if ok:
                    hyps.append(Hypothesis("conv-config", name, z, params, ("conv", name))) # offset: from the stream's start
                    if z > decisive and k/n < 0.1*base:
                        decided = rate
    return hyps, trials

def decode_conv(bits, hyp):
    """Run the config definition behind a conv-config hypothesis on a whole bit stream"""
    from .... import code as coding
    code = coding.load("/".join(hyp.coding))
    seg = np.asarray(bits[hyp.params["offset"]:], dtype=np.uint8)
    nb = puncture_period(code)
    seg = seg[:len(seg)//nb*nb]
    return code.decoder.decode(seg)

def guess_conv(hyp):
    """Definition from a conv hypothesis (generators found by the parity relation, plain or with a scrambler)"""
    p = hyp.params
    gens = [list(p["g"][i]) for i in sorted(p["g"])]
    n = p["n"]
    if len(gens) < n: # Pair relations only name two of the generators
        return None
    from ....code.serial import mother
    m = mother(gens, p["K"])
    obj = {"base": "conv", "extends": m, "comment": f"found by rpps.analysis.identify: the mother code {m}"} if m else \
          {"base": "conv", "constraint": p["K"], "generator": gens}
    return Guess("conv", obj, {"z": hyp.z, "offset": p["offset"]})
