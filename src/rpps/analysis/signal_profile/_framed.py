"""Frames found on timed symbols by a shipped profile, and a preamble's offset (rpps.analysis.signal_profile)"""
import json
from collections import Counter

from ...profile.catalogue import lineage

import numpy as np

VERIFIED = 2 # Frames a step's check must pass (and pass a tenth of the time or more) for the step to count as verified
REFUTED = 3 # Frames a step's check must fail, passing under a tenth as often, for the step to count against the profile

def checks(messages, est=None):
    """{step id: (passed, failed)} of messages. Those with crc_ok True or False (None: no check); est: a stream's
    estimates, whose packet steps (rpps.link.packets) count their lengths meeting the frames' pointers ("aligned") or
    not ("misaligned"): that layer's check"""
    out = {}
    for m in messages:
        if m.crc_ok is None:
            continue
        p, f = out.get(m.kind, (0, 0))
        out[m.kind] = (p + bool(m.crc_ok), f + (not m.crc_ok))
    for k, v in (est or {}).items():
        if isinstance(v, dict) and "aligned" in v and (v["aligned"] or v.get("misaligned")):
            p, f = out.get(k, (0, 0))
            out[k] = (p + int(v["aligned"]), f + int(v.get("misaligned", 0)))
    return out

def evidence(chk):
    """(verified, refuted) step ids of `checks`. Verified: its check passed VERIFIED times and in a tenth of the frames
    or more (a check passing by chance is rare: a CRC-8 1 in 256, so weak symbols that fail most frames still verify
    on the ones they pass); refuted when it failed REFUTED times and passed in under a tenth of them (frames this profile
    reads that aren't this profile's)"""
    ver = sorted(k for k, (p, f) in chk.items() if p >= VERIFIED and 10*p >= p + f)
    ref = sorted(k for k, (p, f) in chk.items() if f >= REFUTED and 10*p < p + f)
    return ver, ref

def rank(found):
    """framed()'s rows, best first. Most steps verified by their own checks, fewest refuted, a payload decoded,
    most headers read; then the most general profile (the fewest it extends): one that only adds layers nothing verified
    is no better a name than the profile it extends"""
    def key(t):
        name, heads, n, _, _, ids, ev = t
        payload = bool(ids) and any(h in heads for h in ids[-1])
        return (-len(ev["verified"]), len(ev["refuted"]), not payload, -len(heads), len(lineage(name)), -n)
    return sorted(found, key=key)

def ties(found):
    """rank()'s rows tied with the first. On everything but how general they are: names the evidence can't tell apart"""
    if not found:
        return []
    def key(t):
        name, heads, n, _, _, ids, ev = t
        payload = bool(ids) and any(h in heads for h in ids[-1])
        return (len(ev["verified"]), len(ev["refuted"]), payload, len(heads))
    k0 = key(found[0])
    return [t for t in found if key(t) == k0]

def framed(raw, det, skip=6000, max_symbols=400000, min_headers=3):
    """Shipped profiles with a frame sync step, run on timed symbols (the frequency offset taken off), conjugated first
    when the preamble turns the wrong way (rpps.sync.inversion.sense: a spectrum inverted on the way down, in which the coherent frame
    sync finds nothing): [(name,
    {header id: ([(fields, count)] commonest first, how many)}, headers, inverted, params, ids, evidence)] of those whose
    headers decode, ids: the profile's header step ids in order, a list each (the headers of a branch point's branches
    together): one of the last decoded says the frames' payload decoded too; evidence: {"verified": step ids whose own
    check (CRC, FCS, a header code) passed, "refuted": step ids whose check kept failing, "checks": {step id: (passed,
    failed)}} (see `evidence`), ranked by `rank`. Each variant a profile's choices make (rp.profile.variants) is tried,
    `params` the one that decoded, of the profiles whose "requires" admits the symbol rate (rp.profile.candidates)"""
    from ... import profile, symbol
    from ...sync.inversion import sense
    out = []
    z = _symbols(raw, det, skip, max_symbols)
    if z is None:
        return out
    # Only the profiles a signal at this rate can be ("requires"): one whose rate is set elsewhere is never built. Not by
    # length: min_samples is the longest frame's, and short frames show their headers in less
    for name, params in profile.candidates(baud=det.baud, input="baseband"):
        try:
            prof = profile.load(name, **params)
        except Exception: # One candidate that doesn't build isn't a reason to lose the others' frames
            continue
        header = next((s.get("header") for s in prof.obj["steps"] if s.get("sync") == "frame"), None)
        if header is None: # No frame sync, or one by a word (no header to read)
            continue
        pre = symbol.load_header(header).preamble
        inverted = sense(z, pre)
        if inverted is None: # The preamble isn't there either way round
            continue
        zz = np.conj(z) if inverted else z
        # The chain's carrier offset is an M-th power's, which a constellation not M-fold symmetric (APSK) leaves off by
        # a good part of the rate: the rest from the preamble's own phase steps
        w = preamble_offset(zz, pre)
        if w:
            zz = (zz*np.exp(-1j*w*np.arange(len(zz)))).astype(np.complex64)
        got = _run(prof, zz)
        if got is None: # A profile that can't run on these symbols isn't this signal
            continue
        heads, chk, _ = got
        if max((len(v) for v in heads.values()), default=0) >= min_headers:
            summary = {k: ([(json.loads(f), n) for f, n in Counter(v).most_common(4)], len(v)) for k, v in heads.items()}
            ids = _header_ids(prof.obj["steps"])
            ver, ref = evidence(chk)
            out.append((name, summary, max(len(v) for v in heads.values()), inverted, params, ids,
                        {"verified": ver, "refuted": ref, "checks": chk}))
    # Verified steps first: DVB-S2 normal, ts, ule and mis share the PL header, and the layers past it tell them apart
    return rank(out)

def _symbols(raw, det, skip=6000, max_symbols=400000, least=20000):
    """Timed symbols from `skip` on, the carrier offset taken off and scaled to unit power; None under `least` of them"""
    z = np.asarray(raw)[skip:skip + max_symbols]
    if len(z) < least:
        return None
    z = z*np.exp(-2j*np.pi*det.cfo/det.baud*np.arange(len(z)))
    return (z/np.sqrt(np.mean(np.abs(z)**2))).astype(np.complex64)

def _run(prof, z, Fs=None, sps=None):
    """A profile's receive on symbols z (or, with `Fs`, on samples z at that rate, `sps` a symbol: from its first
    step): ({header step id: [its fields as JSON]}, checks(...), produced(...)), None when it fails"""
    from ...base import Data
    from ...base.types import Rep
    data = Data(z, sps=1, flush=True) if Fs is None else Data(z, Fs=Fs, sps=sps or 1, flush=True)
    try:
        rx = prof.rx(start=Rep.SYMBOLS) if Fs is None else prof.rx()
        res = rx(data)
        msgs = rx.take_messages() if hasattr(rx, "take_messages") else []
    except Exception:
        return None
    # A profile ending in branches gives {branch: output}: its trunk publishes in the input's stream, each branch in
    # its own; bursts each their own
    def outs(r): # (a profile ending in branches gives a dict, one of bursts a list)
        if isinstance(r, dict):
            return [o for v in r.values() for o in outs(v)]
        if isinstance(r, (list, tuple)):
            return [o for v in r for o in outs(v)]
        return [r] if hasattr(r, "stream") else []
    streams = {id(s): s for s in [data.stream] + [o.stream for o in outs(res)]}
    heads, est = {}, {}
    own = {s.get("id", "inversion") for s in prof.obj["steps"] if s.get("sync") == "inversion"} # (the profile's own
    for st in streams.values():                                              # inversion step: the symbols are put right here)
        for step, fields, at in (st.control.fields() if st.control is not None else []):
            if step not in own:
                heads.setdefault(step, []).append(json.dumps(fields, sort_keys=True, default=str))
        est.update(st.est)
    return heads, checks(msgs, est), produced(msgs, heads, est)

def produced(msgs, heads, est):
    """The step ids that gave anything: a message, a header read, or packets cut (rpps.link.packets' est)"""
    out = {m.kind for m in msgs} | set(heads)
    out |= {k for k, v in est.items() if isinstance(v, dict) and v.get("packets")}
    return out

def deeper(name, params, raw, det, inverted=False, skip=6000, max_symbols=400000):
    """A profile extending `name` (at any depth: rp.profile.catalogue.lineage) whose added layers verify on the timed
    symbols: (its name, its params, evidence as in framed), or None. A code names the most general profile with its steps
    (identify.match_profile: ccsds/tm_rs for ccsds/tm_packets, whose packet layer the code doesn't show); this runs
    the ones built on it and names the deepest whose own steps' checks pass (a packet layer: its lengths meeting the
    frames' pointers), none of its checks refuted, and every link layer it adds gave something. params: the variant of `name` the code matched (kept where the
    deeper profile has them too)"""
    from ... import _config, profile
    kids = [f"{fam}/{m}" for fam, modes in profile.ls(blind=True).items() for m in modes]
    kids = [k for k in kids if name in lineage(k)]
    if not kids:
        return None
    z = _symbols(raw, det, skip, max_symbols, least=4096)
    if z is None:
        return None
    if inverted:
        z = np.conj(z)
    base = set()
    try:
        base = {s.get("id") for s in profile.load(name, **(params or {})).obj["steps"] if s.get("id")}
    except Exception:
        pass
    best = None
    for kid in kids:
        decl = _config.read(_config.find("profile", kid)).get("params") or {}
        own = {k: v for k, v in (params or {}).items() if k in decl or k in _inherited(kid)}
        try:
            prof = profile.load(kid, **own)
        except Exception:
            continue
        got = _run(prof, z)
        if got is None:
            continue
        ver, ref = evidence(got[1])
        new = [k for k in ver if k not in base]
        added = [st["id"] for st in prof.obj["steps"] if st.get("id") and st.get("link") and st["id"] not in base]
        if not new or ref or any(a not in got[2] for a in added): # (a layer of its own that gave nothing isn't there:
            continue                                               # dvbs/si's sections on transport packets of no table)
        key = (len(new), len(lineage(kid)))
        if best is None or key > best[0]:
            best = (key, kid, own, {"verified": ver, "refuted": ref, "checks": got[1]})
    return None if best is None else best[1:]

def _inherited(name):
    """The params a profile takes from the profiles it extends"""
    from ... import _config
    out = set()
    for parent in lineage(name):
        out |= set(_config.read(_config.find("profile", parent)).get("params") or {})
    return out

def shipped(x, Fs, mid, width, budget=2**22):
    """The shipped profiles of a fixed rate a signal may be (rp.profile.candidates: its box `width` no wider than their
    channel), each run from its first step on its own channel: the capture's samples x (at Fs) shifted by -`mid`,
    filtered to the profile's bandwidth and resampled to its rate (an audio input, FM's, ax25/afsk1200, on the channel's
    real part and on its FM discriminator's output); at most `budget` capture samples, or what the profile's
    min_samples needs. [(name, params, evidence as in framed, with "at" and "width": the band decoded, Hz)] of those
    whose own checks verify, none refuted, the most
    verified first, then the most checks passed, then the most general. How the waveforms the linear chain doesn't take
    are named (FSK and GFSK bursts, pulse position, spread spectrum, OFDM: each profile's own receiver, a cell search,
    a burst's preamble), with nothing kept here per standard; a box may be far narrower than its signal (the survey
    takes a pulse train's centre line for it)"""
    from fractions import Fraction
    from ... import profile
    from ...signal.sample import resample_poly
    from ...wave import FMDemod
    from ...base import Data
    x = np.asarray(x)
    out = []
    for name, params in profile.candidates(bw=width):
        try:
            r = profile.info(name, **params).requires
        except Exception:
            continue
        if r["baud"] is None or r["Fs"] is None or r["bw"] is None or abs(mid) + r["bw"]/2 > Fs/2:
            continue # (an open rate is the linear chain's; a channel the capture doesn't hold can't be tried)
        need = int(min(len(x), max(budget, 2*(r["min_samples"] or 0)*Fs/r["Fs"])))
        c, w = _centre(x[:need], Fs, mid, r["bw"])
        at = mid + c
        if abs(at) + r["bw"]/2 > Fs/2:
            continue
        y = x[:need]*np.exp(-2j*np.pi*at/Fs*np.arange(need))
        Y = np.fft.fft(y)
        Y[np.abs(np.fft.fftfreq(need, 1/Fs)) > r["bw"]/2] = 0
        y = np.fft.ifft(Y).astype(np.complex64)
        f = Fraction(float(r["Fs"])/Fs).limit_denominator(1000)
        if f.numerator < 1 or abs(float(f)*Fs/r["Fs"] - 1) > 1e-3:
            continue
        try:
            prof = profile.load(name, **params)
        except Exception:
            continue
        inputs = [y]
        if r["input"] == "audio":
            fm = np.asarray((Data(y, Fs=Fs, flush=True) + FMDemod(gain=1.0)).data, np.float32)
            inputs = [y.real.astype(np.float32), fm]
        for v in inputs:
            v = resample_poly(v, f.numerator, f.denominator) if f != 1 else v
            got = _run(prof, v, Fs=float(r["Fs"]), sps=float(r["Fs"])/float(r["baud"]))
            if got is None:
                continue
            ver, ref = evidence(got[1])
            if ver and not ref:
                out.append((name, params, {"verified": ver, "refuted": ref, "checks": got[1], "at": at,
                                           "width": w or width}))
                break
    def key(t):
        n = sum(p for p, _ in t[2]["checks"].values())
        return (-len(t[2]["verified"]), -n, len(lineage(t[0])))
    return sorted(out, key=key)

def _centre(x, Fs, mid, bw, bins=256, over=6.0):
    """(where the band holding `mid` is centred, from mid [Hz], within +-bw/2; its width [Hz]): the run of a spectrum's
    bins (`bins` across 2 bw about mid) over the floor by `over` dB that holds mid, or the strongest bin near it; (0.0,
    None) with none. A box the survey made of a line inside a wide signal (an LTE carrier's) is off its centre by up to
    half the signal"""
    n = 1 << int(np.ceil(np.log2(max(64, bins*Fs/(2*bw)))))
    m = len(x)//n
    if m < 4:
        return 0.0, None
    seg = np.asarray(x[:m*n]).reshape(m, n)*np.exp(-2j*np.pi*mid/Fs*np.arange(m*n)).reshape(m, n)*np.hanning(n)
    P = np.fft.fftshift(np.mean(np.abs(np.fft.fft(seg, axis=1))**2, axis=0))
    f = np.fft.fftshift(np.fft.fftfreq(n, 1/Fs))
    near = np.abs(f) <= bw
    if near.sum() < 8:
        return 0.0, None
    db = 10*np.log10(P[near] + 1e-30)
    fn = f[near]
    on = db > np.percentile(10*np.log10(P + 1e-30), 10) + over
    k = int(np.argmin(np.abs(fn)))
    if not on[k]:
        cand = np.flatnonzero(on & (np.abs(fn) <= bw/2))
        if not len(cand):
            return 0.0, None
        k = int(cand[np.argmax(db[cand])])
    a = b = k
    while a > 0 and on[a - 1]:
        a -= 1
    while b < len(on) - 1 and on[b + 1]:
        b += 1
    c = 0.5*(fn[a] + fn[b])
    return float(np.clip(c, -bw/2, bw/2)), float(min(fn[b] - fn[a] + (fn[1] - fn[0]), bw))

def preamble_offset(z, preamble, threshold=0.6, top=16, sigmas=3.0):
    """The carrier offset left on symbols z, in radians a symbol (within +-pi), from where a known preamble is: the phase
    steps symbol to symbol, the preamble's own taken off, are the offset's (differential: the carrier phase doesn't
    matter). For symbols the preamble was found in (rpps.sync.inversion.sense); 0.0 when nothing correlates past `threshold`, and when
    the offset is under `sigmas` standard errors of the preambles' own estimates (no offset to see: a noise estimate,
    applied, drifts a long frame's phase by radians)"""
    pre = np.asarray(preamble, dtype=complex)
    ref = pre[1:]*np.conj(pre[:-1])
    d = np.asarray(z[1:])*np.conj(np.asarray(z[:-1]))
    d = d/(np.abs(d) + 1e-12)
    up = np.correlate(d, ref, "valid")/len(ref)
    size = np.abs(up)
    if not len(size) or size.max() <= threshold:
        return 0.0
    # (the preambles: near the strongest; data matches the preamble's steps by chance less well, and at any angle)
    peaks = np.nonzero(size > max(threshold, 0.75*size.max()))[0]
    best = peaks[np.argsort(-size[peaks])[:top]]
    w = float(np.angle(np.sum(up[best])))
    if len(best) > 1:
        spread = np.angle(up[best]*np.exp(-1j*w))
        if abs(w) < sigmas*np.sqrt(np.sum(spread**2)/(len(best) - 1)/len(best)):
            return 0.0
    return w

def _header_ids(steps):
    """A profile's header step ids (symbol or link headers) in order: [[id], ...], the headers of a branch point's
    branches one list"""
    out = []
    for s in steps:
        if s.get("id") and (s.get("symbol") == "header" or s.get("link") == "header"):
            out.append([s["id"]])
        br = s.get("branches")
        if br:
            inner = [h for b in (br.values() if isinstance(br, dict) else br)
                     for g in _header_ids(b.get("steps", []) if isinstance(b, dict) else b) for h in g]
            if inner:
                out.append(inner)
    return out

# --- One signal's chain ----------------------------------------------------------------------------------------------
