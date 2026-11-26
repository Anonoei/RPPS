"""What the receive chain says about a signal's profile: where it is over the air, and whose inversion the spectrum
step saw (rpps.analysis.signal_profile)"""
from ._result import FOUND, GUESS, Step

TRUSTED = ("external", "gpsdo", "gps", "ext", "mimo") # Clock sources that discipline the reference (pysdrlib's names)

def receive_steps(p, source):
    """Add the over-the-air `frequency` step after `acquire`, and attribute the `spectrum` step's inversion: ours (the
    receive chain's LO above the signal, put right at the source) or upstream (the transmitter, a transponder)"""
    rx = getattr(source, "context", None)
    put_right = bool(getattr(source, "conjugated", False) or getattr(getattr(source, "stream", None), "rx_conjugated", False))
    det, sig = getattr(p, "det", None), p.signal
    cf = float(getattr(source, "cf", 0.0) or 0.0)
    kinds = [s.kind for s in p.steps]
    if cf and sig is not None and det is not None and "acquire" in kinds:
        f = cf + sig.mid + det.cfo
        clock = getattr(rx, "clock_source", None)
        ppm = float(getattr(rx, "ppm", 0.0) or 0.0)
        disciplined = clock is not None and str(clock).lower() in TRUSTED
        ref = (f"the reference is {clock}" + (f", corrected by {ppm:+g} ppm" if ppm else "")) if clock else \
            (f"the radio's own reference, corrected by {ppm:+g} ppm" if ppm else "the radio's own reference, uncorrected")
        note = (f"{ref}: good to the reference's accuracy" if disciplined else
                f"{ref}: its error (tens of ppm: tens of kHz a GHz, and an LNB's LO more) is in this, and in the symbol rate")
        if rx is not None and rx.chain is not None and rx.chain.lo:
            note += f"; through a {rx.chain.lo/1e6:g} MHz LO ({'above' if rx.chain.invert else 'below'} the signal)"
        step = Step("frequency", f"carrier {f/1e6:.6f} MHz over the air, {det.baud/1e6:.6g} MBd", FOUND if disciplined else GUESS,
                    detail={"freq": f, "baud": det.baud, "clock_source": clock, "ppm": ppm, "disciplined": disciplined},
                    note=note)
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
