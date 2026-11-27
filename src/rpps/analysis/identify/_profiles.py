"""Shipped profiles a detection matches, and the constellation it targets (rpps.analysis.identify)"""
import functools
import json

import numpy as np

from ._streams import diff_alphabets

def _transform(points, src, dst, rotate=True):
    """(angle, conj) that moves symbols decoded right by labels `src` to where labels `dst` decode them right, or None

    A received symbol near point p carries src[p]; after y -> conj?(y)*e^(j angle) it is near q[p], which dst labels
    dst[q[p]]: the transform where those agree for every point. rotate False: inversion only (phase changes)."""
    from ...symbol.mapping import symmetries
    src, dst = np.asarray(src), np.asarray(dst)
    for th, flip, q in symmetries(points):
        if not rotate and abs(np.angle(np.exp(1j*th))) > 1e-6:
            continue
        if np.array_equal(dst[q], src):
            return float(th), bool(flip)
    return None


def _fec_steps(steps):
    """The code steps that are codes: fec (type, name), and a line code or a sync marker, which a find never has (ax25/afsk1200
    is HDLC behind NRZI, not the HDLC a find names), with their args (an interleaver's depth: a variant of the profile).
    Scramblers and alignment are not compared: a find leaves those to the profile"""
    return [(s["code"], s.get("name"), json.dumps(s.get("args") or {}, sort_keys=True, default=str))
            for s in steps if s.get("code") in ("block", "conv", "interleave", "line", "marker")]


def _symbol_steps(steps):
    """(differential?, constellation name) of a profile's symbol steps"""
    diff = any(s.get("symbol") == "differential" for s in steps)
    cons = [s for s in steps if s.get("symbol") == "constellation"]
    from ...profile.steps import constellation
    return diff, [constellation(s).name for s in cons], cons


def _profile_codes():
    """The names of the convolutional codes the profiles the blind tools try decode with"""
    from ... import _config
    return _codes_of(_config.state())

@functools.lru_cache(maxsize=None)
def _codes_of(state):
    out = set()
    for name, obj, _ in _profiles():
        out |= {st["name"] for st in obj.get("steps", []) if st.get("code") == "conv" and isinstance(st.get("name"), str)}
    return frozenset(out)


def _profiles(baud=None):
    """(name, resolved definition, params) of every shipped profile and each variant its choices make (rp.profile.variants);
    baud: only those a signal at that symbol rate can be (rp.profile.candidates)"""
    from ... import _config
    from ...profile.catalogue import candidates
    for name, params in candidates(baud=baud):
        try:
            yield name, _config.resolve(_config.read(_config.find("profile", name)), params, f"profile/{name}"), params
        except Exception:
            continue


PROBE = 900001 # A value no param has: an arg that comes out as it (or PROBE + k) is that param's

def _bind(name, params, steps):
    """(resolved definition, params) of profile `name` with the params its code steps' args come from set as `steps`
    has them (ccsds/tm_rs's interleave depth from a find at depth 1), or None when none do"""
    from ... import _config
    try:
        raw = _config.read(_config.find("profile", name))
        free = [k for k, v in (raw.get("params") or {}).items()
                if k not in params and isinstance(v, dict) and isinstance(v.get("default"), int)
                and not isinstance(v.get("default"), bool)]
        if not free:
            return None
        probe = {k: PROBE + i for i, k in enumerate(free)}
        mine = [s for s in _config.resolve(raw, {**params, **probe}, f"profile/{name}")["steps"]
                if s.get("code") in ("block", "conv", "interleave", "line", "marker")]
        theirs = [s for s in steps if s.get("code") in ("block", "conv", "interleave", "line", "marker")]
        if len(mine) != len(theirs):
            return None
        set_ = {}
        for a, b in zip(mine, theirs):
            for k, v in (a.get("args") or {}).items():
                if isinstance(v, int) and PROBE <= v < PROBE + len(free):
                    set_[free[v - PROBE]] = (b.get("args") or {}).get(k)
        if not set_ or any(v is None for v in set_.values()):
            return None
        params = {**params, **set_}
        return _config.resolve(raw, params, f"profile/{name}"), params
    except Exception:
        return None

def match_profile(steps, baud=None):
    """A shipped profile (config/profile), or a variant of one, with the same code and symbol steps as `steps`, the most
    general when several do (rp.profile.catalogue.lineage): (name, its steps, the params of the variant: {} for the
    defaults) or (None, None, None). baud: the symbol rate,
    when known: a profile whose "requires" sets another isn't named"""
    from ...profile.catalogue import lineage
    want = (_fec_steps(steps), _symbol_steps(steps)[:2])
    hits = []
    for name, obj, params in _profiles(baud):
        try:
            got = (_fec_steps(obj["steps"]), _symbol_steps(obj["steps"])[:2])
        except Exception: # A profile whose steps a header names (no fixed code) can't match
            continue
        if got != want and got[1] == want[1] and [g[:2] for g in got[0]] == [w[:2] for w in want[0]]:
            # The same codes, other args: a param of the profile's (an interleaver's depth) may set them as found
            bound = _bind(name, params, steps)
            if bound is not None:
                obj, params = bound
                got = (_fec_steps(obj["steps"]), _symbol_steps(obj["steps"])[:2])
        if got == want:
            hits.append((len(lineage(name)), len(hits), name, obj["steps"], params))
    # The most general of those that match: ccsds/tm_packets extends ccsds/tm_rs with a link layer the code doesn't show
    return min(hits)[2:] if hits else (None, None, None)


def _finish(out, c, profiles, baud=None):
    """The shipped profile with the same steps (one a signal at `baud` can be), if any, and the correction that turns the symbols to the map it decodes with"""
    own = out.steps
    name, steps, params = match_profile(own, baud) if profiles else (None, None, None)
    for out.profile_name, out.profile_params, out.steps in ((name, params, steps), (None, None, own)):
        if out.steps is None:
            continue
        out.target, out.correction = _target(out, c)
        if out.correction is not None:
            out.inverted = _inverted(out, c) if out.profile_name else None
            return out
    out.found = False # (no turn of these symbols gives the map their own steps name: can't happen)
    return out


def _inverted(out, c):
    """Whether the symbols are spectrally inverted, by a standard's labelling: a shipped profile's map, which the code
    decodes on only conjugated (True) or only as received (False). None when the streams that decode (the one found and
    its equivalents: a code that can't tell inverted bits apart) need both senses"""
    flips = set()
    for key in [out.stream, *out.equivalent]:
        t = _target(out, c, key)[1]
        if t is not None:
            flips.add(t[1])
    return flips.pop() if len(flips) == 1 else None


def _target(out, c, stream=None):
    """(the map the steps decode with, the correction that turns the symbols to it, or None); stream: the bit stream
    whose labels to start from (default the one found)"""
    stream = out.stream if stream is None else stream
    spec = _symbol_steps(out.steps)[2][0]
    from ...profile.steps import constellation
    target = constellation(spec).mapping
    if stream[0] == "diff": # A phase change sees no turn of the carrier, only the IQ sense
        alph = diff_alphabets(c)[stream[1]]
        return target, _transform(alph.points, alph.maps[stream[2]].arr, target.arr, rotate=False)
    t = _transform(c._base, c.blind_maps[stream[1]].arr, target.arr)
    if t is None or not target.turn:
        return target, t
    return target, ((t[0] + target.turn) % (2*np.pi), t[1]) # Then onto the points the target's map turns them to
