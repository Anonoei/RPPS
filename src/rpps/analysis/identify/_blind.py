"""The blind tools when no definition fits: solve, and the tiers identify runs (rpps.analysis.identify)"""
import numpy as np

from ...code.coding import puncture_period
from ..result import Hypothesis, Report
from ..code.conv.mother import guess_conv
from ..framing.bits import scan_hdlc
from ..sequence.scrambler import cached_scrambler
from ..sequence import scrambler
from ..code.conv import puncture
from ..code.block import rs as blind_rs, linear as blind_block
from ...code.scram.scrambler import Feedthrough

from ._scans import detect_fec, scan_rs

def solve(bits, rs_lengths=None, max_taps=2, exhaustive=False, block=True):
    """Peel unknown layers off bits (a decoder's output): [(hypothesis, Guess or None)], outermost first

    1. Reed-Solomon on the bits as they are: a configured one (scan_rs, a Forney interleaver too), else a blind one
       (rpps.analysis.code.block.rs.search over `rs_lengths`, or every length with `exhaustive`)
    2. a binary block code (rpps.analysis.code.block.linear.solve_block: BCH / cyclic, product, LDPC or any linear code, by rank)
    3. a feedthrough scrambler (rpps.analysis.sequence.scrambler.search, up to `max_taps` taps); if one is found, the same codes on
       what it gives, then HDLC framing. A scrambler hypothesis with no taps means the bits already had structure.
    4. none found (descrambled data that is random itself, compressed or encrypted, stays white): Reed-Solomon behind each
       configured feedthrough scrambler (its structure is algebraic: a scrambler search can't see it)
    Nothing here decodes: a Guess is a definition once it is saved. Empty: nothing found (the bits stay white: another
    layer, or random payload such as encrypted data)."""

    def code_layer(b, with_block=block):
        h, t, _ = scan_rs(b)
        v = Report(h, t, None).verdict
        if v is not None:
            return [(v, None)]
        h, t, g = blind_rs.search(b, lengths=rs_lengths, exhaustive=exhaustive)
        if h:
            return [(h[0], g)]
        return blind_block.solve_block(b) if with_block else []

    bits = np.asarray(bits, dtype=np.uint8)
    found = code_layer(bits)
    if found:
        return found
    # The whole stream an LFSR sequence (a PRBS, or scramblers over repeating data): that explains every bit, and a
    # feedthrough search would only find a part of it
    ha, ta, ga = scrambler.additive_search(bits)
    if ha and ha[0].params["holds"] > 0.95:
        return [(ha[0], ga)]
    h, t, g = scrambler.search(bits, max_taps=max_taps)
    if not h and ha: # Not a feedthrough scrambler: an additive one showing through constant stretches
        return [(ha[0], ga)]
    if not h: # Bits that stay white: maybe RS-coded random payload behind a known scrambler (RS structure is algebraic)
        from ... import code
        for name in code.scram.ls(blind=True).get("feedthrough", []):
            obj = cached_scrambler(f"feedthrough/{name}")
            found = code_layer(Feedthrough.descramble(bits, obj["poly"]), with_block=False)
            if found:
                return [(Hypothesis("scrambler", f"feedthrough/{name}", found[0][0].z, {"taps": list(obj["poly"]), "config": name}),
                         None)] + found
        return []
    out = [(h[0], g)]
    taps = h[0].params["taps"]
    d = Feedthrough.descramble(bits, taps) if taps else bits
    if taps:
        found = code_layer(d)
        if found:
            return out + found
    hd = Report(*scan_hdlc({"bits": d}), None).verdict
    if hd is not None:
        out.append((hd, None))
    return out


def _llrs(y, c, head):
    """LLRs of y under each coherent map of head (the soft side of bit_streams: what the configured scans score softly)"""
    from ... import symbol
    out = {}
    for key, bits in head.items():
        if key[0] != "abs":
            continue
        try:
            m = c.blind_maps[key[1]].copy()
            m.turn = 0.0 # (its labels on the points as the symbols sit: the turn is a carrier phase here)
            out[key] = symbol.load_or_generate(c.name, mapping=m).llr(y)[:len(bits)]
        except Exception: # A map the constellation can't take by index: that stream is scored hard only
            continue
    return out

def _distinct(head, c):
    """The coherent bit streams that differ by more than inverted bits: maps whose turn is below 180 degrees"""
    keys = [k for k in head if k[0] == "abs" and np.angle(np.exp(1j*c.blind_maps[k[1]].rotation)) >= -1e-9
            and np.angle(np.exp(1j*c.blind_maps[k[1]].rotation)) < np.pi - 1e-9]
    return {k: head[k] for k in keys} or {k: v for k, v in head.items() if k[0] == "abs"}


def _blind(head, c, y=None):
    """With no configured code: a known mother code with a puncture pattern none of them has (then what is inside it), or
    a binary block code, on the channel bits (y: the symbols, for soft values). [(hypothesis, Guess or None)], outermost
    first, or []"""
    hyps, trials = puncture.search(head)
    v, g = puncture.best(hyps, trials)
    if v is not None:
        code = puncture._code(v.params["K"], v.params["generator"], np.asarray(v.params["puncture"], dtype=bool))
        bits = np.asarray(head[v.params["stream"]][v.params["offset"]:], dtype=np.uint8)
        m = puncture_period(code)
        info = code.decoder.decode(bits[:len(bits)//m*m]).astype(np.uint8)
        return [(v, g)] + solve(info)
    # A convolutional code no definition has, before block codes: a convolutional code is a linear code over any block
    # length too, and the block search would name it one (scan_conv relabels short periodic checks as linear-block)
    sub = {k: v for k, v in head.items() if k[0] == "abs"}
    cv = detect_fec(sub, blocks=False, max_offsets=32).verdict
    if cv is not None and cv.kind == "conv":
        g = guess_conv(cv)
        if g is not None:
            from ...code import Conv
            p = cv.params
            gens = np.asarray([p["g"][i] for i in sorted(p["g"])], dtype=np.uint8).reshape(p["n"], p["K"])
            bits = np.asarray(head[p["stream"]][p["offset"]:], dtype=np.uint8)
            info = Conv(gens).decoder.decode(bits[:len(bits)//p["n"]*p["n"]]).astype(np.uint8)
            return [(cv, g)] + solve(info)
    # Every map: one that swaps the bits within a symbol cuts across codewords of odd length, and can still show weaker
    # structure at a multiple of it, so the shortest code (then the most checks) of all the maps is the one
    best = None
    for key, bits in _distinct(head, c).items():
        llr = None
        if y is not None: # Soft values on this map: the cleanest words are the ones a sparse check search uses
            from ... import symbol
            cc = symbol.load_or_generate(c.name)
            cc.set_mapping(key[1])
            llr = cc.llr(y)[:len(bits)]
        found = blind_block.solve_block(bits, llr=llr)
        if found:
            for h, _ in found:
                h.params["stream"] = key
            n, k = found[0][0].params.get("n", 1 << 30), found[0][0].params.get("k", 0)
            if best is None or (n, k) < best[0]:
                best = ((n, k), found)
    return best[1] if best else []
