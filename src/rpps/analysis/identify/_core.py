"""identify: which code a symbol stream carries (rpps.analysis.identify)"""
import numpy as np

from ...code.coding import puncture_period, cached
from ..result import Hypothesis, Report, pick
from ..code.conv.mother import scan_defined_conv, guess_conv
from ..code.block.linear import recover_dual
from ..framing.bits import frame_markers, MARKER_PERIODS, scan_hdlc
from ..framing.marked import layer
from ..code.block import linear as blind_block

from ._blind import _blind, _llrs, solve
from ._detection import CodeDetection
from ._profiles import _finish, _profile_codes
from ._scans import config_blocks, decode_tail, detect_fec, guess_block, scan_block, scan_rs
from ._streams import CONV_SYMBOLS, _constellation, bit_streams, diff_alphabets, repeat_period

def identify(symbols, constellation, names=None, blind=False, differential=True, profiles=True, baud=None, prior=None):
    """Which code from config/fec timed, carrier-corrected symbols carry, under which map, and how to decode it

    In turn, stopping at the first that stands out: configured convolutional codes (any puncture phase), with a Reed-Solomon
    code behind them (scan_rs; then the markers that start its codewords, for Align); HDLC framing; configured block codes.
    blind: look for what no definition covers, as `Guess`es only (nothing decodes from a guess). With no configured code: a
        known mother code with an unknown puncture pattern (rpps.analysis.code.conv.puncture) and what is inside it, a binary block code
        on the channel bits (rpps.analysis.code.block.linear), then unknown rate 1/n convolutional codes (detect_fec). Inside a
        convolutional code with no configured RS behind it: `solve` (RS, block codes, scramblers). Tens of seconds.
    symbols: the symbols (any gain); constellation: a Constellation or its name (the detected modulation)
    names: only these convolutional definitions. baud: the symbol rate, when known: the shipped profile named
    (profile_name) is one whose "requires" admits it (rp.profile.fits). Returns a CodeDetection
    A stream that repeats (`repeat_period`: fill, a test pattern) has the period in `repeat`, and a code found on it
    stands only when one period shows it too.
    prior: what identify gave on the start of these same symbols (AcquireFEC's previous, shorter window). The configured
    scans look at the first CONV_SYMBOLS only: when prior saw all of them, its answer there stands (a convolutional code
    found is decoded again on the longer window for what is inside it; nothing found goes straight to the blind tools)
    """
    y = np.asarray(symbols, dtype=np.complex128)
    P = repeat_period(y) if len(y) >= 1024 else None
    out = _identify(y, constellation, names, blind, differential, profiles, baud, prior)
    if P is not None:
        # A code over copies is only as good as it is on one: the layers inside it (RS behind a convolutional code over
        # repeated packets) still use every copy
        if out.found and not _identify(y[:P], constellation, names, False, differential, False, baud, None).found:
            out = _identify(y[:P], constellation, names, blind, differential, profiles, baud, None)
        out.repeat, out.complete = P, True # More symbols are more copies
    return out

def _identify(symbols, constellation, names, blind, differential, profiles, baud, prior):
    c = _constellation(constellation)
    y = np.asarray(symbols, dtype=np.complex128)
    if len(y):
        y = y/np.sqrt(np.mean(np.abs(y)**2))
    out = CodeDetection(mod=c.name, n=len(y))
    if len(y) < 1024:
        return out
    head = bit_streams(y[:CONV_SYMBOLS], c, differential)
    pre = [{"sync": "timing"}]

    def symbol_steps(key):
        if key[0] == "diff":
            d = diff_alphabets(c)[key[1]]
            return [{"symbol": "differential"}, {"symbol": "constellation", "name": key[1], "mapping": d.maps[key[2]].comment}]
        from ... import _config
        turned = c.turned_name(c.blind_maps[key[1]]) # A turned map's labels: the step names it (the correction turns the symbols)
        return [{"symbol": "constellation", **({"name": c.name} if _config.find("constellation", c.name) else {"shape": c.name}),
                 **({"mapping": turned} if turned else {})}]

    tried = [] # Every tier's hypotheses: the report when nothing stands out
    same = prior is not None and prior.n is not None and prior.n >= CONV_SYMBOLS and len(y) >= CONV_SYMBOLS \
        and prior.report is not None # The configured scans would see what prior's saw
    if same and not prior.found:
        out.report, out.complete = prior.report, True # Nothing configured there: on to the blind tools, if any
        return _blind_tiers(out, head, c, y) if blind else out
    if same and prior.kind == "conv":
        rep = prior.report # (the convolutional tier's own report: the pick below makes the same choice)
    else:
        hyps, trials = scan_defined_conv(head, names, llrs=_llrs(y[:CONV_SYMBOLS], c, head))
        rep = Report(hyps, trials, None)
    tried.append(rep)
    conv = pick(rep.hyps, rep.z_crit, lambda h: h.params["mismatch"]/h.params["baseline"],
                 lambda b, h: h.params["mismatch"] < 0.5*(b.params["mismatch"] + b.params["baseline"]))
    if conv is not None and profiles:
        # Codes that are one another under a different map (a rate 1/2 code with its generators swapped is the same code
        # with I and Q swapped) fit as well, and which one wins is chance: prefer the one a shipped profile decodes with
        shipped = _profile_codes()
        if conv.coding[1] not in shipped:
            alike = [h for h in rep.hyps if h.coding[1] in shipped and h.z > rep.z_crit
                     and h.params["mismatch"] < 0.5*(conv.params["mismatch"] + conv.params["baseline"])]
            if alike:
                conv = min(alike, key=lambda h: h.params["mismatch"]/h.params["baseline"])
    if conv is not None:
        key = conv.params["stream"]
        same = [h for h in rep.hyps if h.coding == conv.coding and h.params["stream"] != key and h.z > rep.z_crit
                and h.params["mismatch"] < 0.5*(conv.params["mismatch"] + conv.params["baseline"])]
        same = list({str(h.params["stream"]): h for h in same}.values())
        out = CodeDetection(found=True, mod=c.name, kind="conv", name=conv.coding[1], stream=key, offset=conv.params["offset"],
                            period=puncture_period(cached("/".join(conv.coding))),
                            equivalent=[h.params["stream"] for h in same], z=conv.z, mismatch=conv.params["mismatch"],
                            baseline=conv.params["baseline"], report=rep, n=len(y))
        steps = pre + symbol_steps(key) + [{"sync": "puncture", "conv": out.name},
                                           {"code": "conv", "name": out.name}]
        info = decode_tail(bit_streams(y, c, differential, keys=[key])[key], cached("/".join(conv.coding)))
        h, t, short = scan_rs(info)
        rs_rep = Report(h, t, None)
        rs = rs_rep.verdict
        out.complete = rs is not None or not short # More symbols could still show a Reed-Solomon code inside
        fl = layer(info) if rs is None else None # Frames behind a marker, the code inside them (CCSDS TM concatenated)
        if rs is None and (fl is None or fl.code is None): # Long frames (10232 bits at depth 5) want more than the tail
            more = decode_tail(bit_streams(y, c, differential, keys=[key])[key], cached("/".join(conv.coding)), fraction=0.95)
            if len(more) > 1.3*len(info):
                fl = layer(more)
        if fl is not None and fl.code is not None:
            out.inner.append(Hypothesis("frame", fl.code, fl.z, {"marker": fl.marker, "period": fl.period,
                                        "scrambler": fl.scrambler, "interleaver": fl.interleaver, "depth": fl.depth,
                                        "words_ok": fl.ok/fl.words}, ("block", fl.code)))
            out.complete = True
            out.steps = steps + fl.steps
            return _finish(out, c, profiles, baud)
        if rs is None and blind:
            _attach(out, solve(info))
            if not out.inner: # Nothing inside: what the bits are like (random-looking: keyed or compressed, no layer)
                from ..framing.payload import verdict
                out.payload = verdict(info)
        if rs is not None:
            out.inner.append(rs)
            p = rs.params
            n = cached(f"block/{rs.name}").encoder.den
            pat, _ = frame_markers(info ^ p["polarity"], n, p["bit_offset"], p["byte_phase"])
            if pat is None and len(info)//(n*8) < MARKER_PERIODS:
                out.complete = False # Too few codewords to see the markers that start them: more symbols may show them
            if pat is not None:
                p["pattern"] = pat
                steps += [{"sync": "align", "period": n, "pattern": pat, "name": "packets"}]
                if p["interleaver"]:
                    steps += [{"code": "interleave", "name": p["interleaver"]}]
                steps += [{"code": "block", "name": rs.name}]
        out.steps = steps
        return _finish(out, c, profiles, baud)

    found = _framed_layer(y, c, differential)
    if found is not None:
        key, fl = found
        rep = Report([Hypothesis("frame", fl.code, fl.z, {"stream": key, "marker": fl.marker, "period": fl.period,
                                                          "words_ok": fl.ok/fl.words}, ("block", fl.code))], 1, None)
        out = CodeDetection(found=True, complete=True, mod=c.name, kind="block", name=fl.code, stream=key,
                            offset=fl.first, period=fl.period, z=fl.z, report=rep, n=len(y))
        out.steps = pre + symbol_steps(key) + fl.steps
        return _finish(out, c, profiles, baud)

    hyps, trials = scan_hdlc(head)
    hyps = [h for h in hyps if h.params["polarity"] == 0] # Another map gives the other polarity: the profile has none
    rep = Report(hyps, trials, None)
    tried.append(rep)
    if rep.verdict is not None:
        v = rep.verdict
        out = CodeDetection(found=True, complete=True, mod=c.name, kind="hdlc", name=v.coding[1], stream=v.params["stream"],
                            offset=0, period=None, z=v.z, report=rep, n=len(y))
        out.steps = pre + symbol_steps(out.stream) + [{"code": "uncoded"}, {"link": "frame", "name": out.name}]
        return _finish(out, c, profiles, baud)

    hyps, trials = [], 0
    blocks = config_blocks()
    for key, bits in head.items():
        for name, folder, cfg, bn, r, fails in blocks:
            h, t = scan_block(bits, fails, r, bn, name, key, max_offsets=bn)
            for hh in h:
                hh.coding = (folder, cfg)
            hyps += h
            trials += t
    rep = Report(hyps, trials, None)
    v = pick(rep.hyps, rep.z_crit, lambda h: (-h.params["words_ok"], -h.params["n"]),
              lambda b, h: h.params["words_ok"] >= 0.5*b.params["words_ok"])
    if v is not None:
        out = CodeDetection(found=True, complete=True, mod=c.name, kind="block", name=v.coding[1], stream=v.params["stream"],
                            offset=v.params["offset"], period=v.params["n"], z=v.z, report=rep, n=len(y))
        out.steps = pre + symbol_steps(out.stream) + [{"code": "block", "name": out.name}]
        if blind: # Its words may be the rows of a product code (a turbo product code): columns of them that are codewords too
            bits, n1 = head[out.stream], out.period
            p = blind_block.product(bits, n1, out.offset)
            found = None
            if p is not None:
                rows = recover_dual(bits, n1, out.offset, trials=20)
                found = blind_block._product(bits, n1, rows["k"], rows["H"], p[0], out.offset + p[1]*n1)
            if found is not None:
                _attach(out, [found])
        return _finish(out, c, profiles, baud)

    tried.append(rep)
    out.report = Report([h for r in tried for h in r.hyps], sum(r.trials for r in tried), None)
    out.complete = len(y) >= CONV_SYMBOLS # The scans above look no further: more symbols would not change this
    return _blind_tiers(out, head, c, y) if blind else out


def _framed_layer(y, c, differential):
    """(stream key, Frames) of the first bit stream of y whose frames behind a sync marker carry a code that decodes
    (rpps.analysis.framing.marked), the streams in bit_streams' order, a marker found in its own polarity first"""
    streams = bit_streams(y, c, differential)
    found = []
    for key, bits in streams.items():
        fl = layer(bits)
        if fl is not None and fl.code is not None:
            found.append((fl.inverted, -fl.ok/fl.words, len(found), key, fl))
            if not fl.inverted:
                break
    return min(found)[3:] if found else None


def _blind_tiers(out, head, c, y):
    """identify's blind tools, when no configured code fits: a mother code's unknown puncture, a block code, an unknown
    convolutional code (_blind), else detect_fec's guess"""
    found = _blind(head, c, y[:CONV_SYMBOLS])
    if found:
        _attach(out, found)
        return out
    sub = {k: v for k, v in head.items() if k[0] == "abs"}
    brep = detect_fec(sub, max_offsets=32)
    out.report = brep
    v = brep.verdict
    if v is not None:
        g = None
        if v.kind.startswith("conv"):
            g = guess_conv(v)
        elif v.kind == "linear-block":
            g = guess_block(sub[v.params["stream"]], hints=[int(round(2*v.params["period_pairs"]))])
        out.guesses = [g] if g is not None else []
    return out


def _attach(out, found):
    """Add [(hypothesis, Guess or None)] to a detection: each hypothesis as an inner layer, carrying its guess"""
    for h, g in found:
        h.guess = g
        out.inner.append(h)
        if g is not None:
            out.guesses.append(g)
