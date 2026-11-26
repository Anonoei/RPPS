"""Frames found on timed symbols by a shipped profile, and a preamble's offset (rpps.analysis.signal_profile)"""
import json
from collections import Counter

import numpy as np

def framed(raw, det, skip=6000, max_symbols=400000, min_headers=3):
    """Shipped profiles with a frame sync step, run on timed symbols (the frequency offset taken off), conjugated first
    when the preamble turns the wrong way (rpps.sync.inversion.sense: a spectrum inverted on the way down, in which the coherent frame
    sync finds nothing): [(name,
    {header id: ([(fields, count)] commonest first, how many)}, headers, inverted, params, ids)] of those whose headers decode,
    ids: the profile's header step ids in order, a list each (the headers of a branch point's branches together): one of
    the last decoded says the frames' payload decoded too,
    those whose payload decoded first, then most headers first; each variant a profile's choices make
    (rp.profile.variants) is tried, `params` the one that decoded, of the profiles whose "requires" admits the symbol rate
    (rp.profile.candidates)"""
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
        prof = profile.load(name, **params)
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
            res = prof.rx(start=Rep.SYMBOLS)(data)
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
            out.append((name, summary, max(len(v) for v in heads.values()), inverted, params, ids))
    # A profile whose payload decoded before one whose headers only read (DVB-S2 normal and mis share the PL header)
    return sorted(out, key=lambda t: (not (t[5] and any(h in t[1] for h in t[5][-1])), -t[2]))

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
