"""A signal's profile: every step it takes to process it, as far as analysis can tell, and what it might be where it can't

    profiles = rp.analysis.profile(rp.source.File("ci16t", "capture.cs16", Fs=100e6))   # every signal the survey finds
    profiles = rp.analysis.profile(src, signals=[0, 3])                                   # or some of them
    for p in profiles:
        print(p)                       # numbered steps; an unsolved one ends the list with its top candidates
    p.solved, p.to_dict(), p.rx()      # whether every step is known, a JSON-able form, the receiver (when solved)

Each signal goes through the chain once: its channel (every signal's from one rp.signal.FFTBank: one FFT of the
capture a hop for all of them), Acquire (rate, modulation, offset), the
matrix matched filter and feed-forward carrier, then AcquireFEC (the code, its bit map and alignment, with the blind tools
when `blind`). When no code is found, shipped profiles that find their own frames (a preamble and a header: DVB-S2) are run
on the timed symbols. A step is "found" (a definition covers it), "guess" (a blind find: a definition once saved, see
rpps.analysis.result.Guess) or "unsolved", which ends the profile with what it might be: hypotheses and their weight, a
softmax over their z-scores (a clear leader near 100%, close calls shared).
"""
from ._result import FOUND, GUESS, UNSOLVED, MIN_PAYLOAD, Step, SignalProfile, weights, _guess_steps
from ._framed import framed, preamble_offset, rotation, _header_ids
from ._measures import RATE_WIDTH, STRONG, ESN0_GAP, NEAR, PM_OFFSET, CARRIER_SHARE, SYMS, esn0_gap, carrier_share, _strong
from ._run import profile, _refit
