"""What the receive chain says about a signal's profile: where it is over the air, and whose inversion the spectrum
step saw (rpps.analysis.signal_profile)"""
from ._result import FOUND, GUESS, Step

def _hz(v):
    return f"{v:.0f} Hz" if v < 1e3 else f"{v/1e3:.2f} kHz"

def accuracy(rx, rf):
    """(standard error [Hz] of a carrier read at `rf` over the air, of a rate [ppm], what isn't known) from a receive
    context: the reference's (`ppm_accuracy`, after the correction pysdrlib applied) at the carrier's IF, and the chain's
    LO (`lo_accuracy`). A part not known is None, and named in the third"""
    ppm_acc = getattr(rx, "ppm_accuracy", None)
    chain = getattr(rx, "chain", None)
    lo = chain is not None and bool(chain.lo)
    lo_acc = (getattr(chain, "lo_accuracy", None) if lo else 0.0)
    f_if = abs(chain.to_if(rf)) if lo else abs(rf)
    unknown = [w for w, v in (("the reference's accuracy", ppm_acc), ("the LO's accuracy", lo_acc)) if v is None]
    if unknown:
        return None, ppm_acc, unknown
    return float(((f_if*ppm_acc*1e-6)**2 + lo_acc**2)**0.5), ppm_acc, []

def receive_steps(p, source):
    """Add the over-the-air `frequency` step after `acquire` (with what it is worth, from the receive context's
    accuracies), and attribute the `spectrum` step's inversion: ours (the receive chain's LO above the signal, put right
    at the source) or upstream (the transmitter, a transponder)"""
    rx = getattr(source, "context", None)
    put_right = bool(getattr(source, "conjugated", False) or getattr(getattr(source, "stream", None), "rx_conjugated", False))
    det, sig = getattr(p, "det", None), p.signal
    cf = float(getattr(source, "cf", 0.0) or 0.0)
    kinds = [s.kind for s in p.steps]
    if cf and sig is not None and det is not None and "acquire" in kinds:
        f = cf + sig.mid + det.cfo
        sf, ppm_acc, unknown = accuracy(rx, f)
        ppm = float(getattr(rx, "ppm", 0.0) or 0.0)
        clock = getattr(rx, "clock_source", None)
        ref = (f"reference {clock}" if clock else "the radio's own reference") + (f", corrected by {ppm:+g} ppm" if ppm else "")
        lo = getattr(getattr(rx, "chain", None), "lo", 0.0)
        through = f"; through a {lo/1e6:g} MHz LO ({'above' if rx.chain.invert else 'below'} the signal)" if lo else ""
        if sf is not None:
            sb = det.baud*ppm_acc*1e-6
            step = Step("frequency", f"carrier {f/1e6:.6f} MHz +- {_hz(sf)} over the air, {det.baud/1e6:.7g} MBd +- {sb:.0f} Bd",
                        FOUND, detail={"freq": f, "freq_sigma": sf, "baud": det.baud, "baud_sigma": sb, "ppm": ppm,
                                       "ppm_accuracy": ppm_acc, "clock_source": clock},
                        note=f"{ref}, known to {ppm_acc:g} ppm{through}")
        else:
            step = Step("frequency", f"carrier {f/1e6:.6f} MHz over the air, {det.baud/1e6:.6g} MBd", GUESS,
                        detail={"freq": f, "baud": det.baud, "ppm": ppm, "ppm_accuracy": ppm_acc, "clock_source": clock},
                        note=f"{ref}{through}: {' and '.join(unknown)} not known (pysdrlib set_freq_correction(ppm, accuracy=), a "
                             "chain's lo_accuracy), so neither is what this is worth: tens of ppm is tens of kHz a GHz")
        p.steps.insert(kinds.index("acquire") + 1, step)
    spec = next((s for s in p.steps if s.kind == "spectrum"), None)
    ours = "the receive chain's LO is above the signal: conjugated at the source"
    if spec is None:
        if put_right:
            at = next((i + 1 for i, s in enumerate(p.steps) if s.kind == "channel"), len(p.steps))
            p.steps.insert(at, Step("spectrum", "receive chain inverted, conjugated at the source", FOUND,
                                    note="its LO is above the signal (the recording's chain says so)"))
        return p
    if spec.what.startswith("inverted"):
        if rx is not None: # Our own path is accounted for: what is left is the transmitter's or a transponder's
            spec.what = "inverted upstream (the transmitter or a transponder): " + spec.what[len("inverted: "):]
        else:
            spec.note = (spec.note + "; " if spec.note else "") + "the receive chain isn't known: our own path or upstream"
    if put_right:
        spec.note = (spec.note + "; " if spec.note else "") + ours
    return p
