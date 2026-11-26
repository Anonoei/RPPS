"""Frames found on timed symbols by a shipped profile, and a preamble's offset (rpps.analysis.signal_profile)"""
import json
from collections import Counter

from ...profile.catalogue import lineage

import numpy as np

VERIFIED = 2 # Frames a step's check must pass (and pass at least as often as it fails) for the step to count as verified
REFUTED = 3 # Frames a step's check must fail, passing under a tenth as often, for the step to count against the profile

def checks(messages):
    """{step id: (passed, failed)} of messages. Those with crc_ok True or False (None: no check)"""
    out = {}
    for m in messages:
        if m.crc_ok is None:
            continue
        p, f = out.get(m.kind, (0, 0))
        out[m.kind] = (p + bool(m.crc_ok), f + (not m.crc_ok))
    return out

def evidence(chk):
    """(verified, refuted) step ids of `checks`. Verified: its check passed VERIFIED times and at
    least as often as it failed (a check passing by chance is rare: a CRC-8 1 in 256); refuted when it failed REFUTED
    times and passed under a tenth as often (frames this profile reads that aren't this profile's)"""
    ver = sorted(k for k, (p, f) in chk.items() if p >= VERIFIED and p >= f)
    ref = sorted(k for k, (p, f) in chk.items() if f >= REFUTED and 10*p < f)
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
    from ...base import Data
    from ...sync.inversion import sense
    from ...base.types import Rep
    out = []
    z = np.asarray(raw)[skip:skip + max_symbols]
    if len(z) < 20000:
        return out
    z = z*np.exp(-2j*np.pi*det.cfo/det.baud*np.arange(len(z)))
    z = (z/np.sqrt(np.mean(np.abs(z)**2))).astype(np.complex64)
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
        data = Data(zz, sps=1, flush=True)
        try:
            rx = prof.rx(start=Rep.SYMBOLS)
            res = rx(data)
            msgs = rx.take_messages() if hasattr(rx, "take_messages") else []
        except Exception: # A profile that can't run on these symbols isn't this signal
            continue
        # A profile ending in branches gives {branch: output}: its trunk publishes in the input's stream, each
        # branch in its own
        streams = [data.stream] + [o.stream for o in (res.values() if isinstance(res, dict) else [res])]
        heads = {}
        own = {s.get("id", "inversion") for s in prof.obj["steps"] if s.get("sync") == "inversion"} # (the profile's own
        for st in {id(s): s for s in streams}.values():                         # inversion step: the symbols are put right here)
            for step, fields, at in (st.control.fields() if st.control is not None else []):
                if step not in own:
                    heads.setdefault(step, []).append(json.dumps(fields, sort_keys=True, default=str))
        if max((len(v) for v in heads.values()), default=0) >= min_headers:
            summary = {k: ([(json.loads(f), n) for f, n in Counter(v).most_common(4)], len(v)) for k, v in heads.items()}
            ids = _header_ids(prof.obj["steps"])
            chk = checks(msgs)
            ver, ref = evidence(chk)
            out.append((name, summary, max(len(v) for v in heads.values()), inverted, params, ids,
                        {"verified": ver, "refuted": ref, "checks": chk}))
    # Verified steps first: DVB-S2 normal, ts, ule and mis share the PL header, and the layers past it tell them apart
    return rank(out)

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
