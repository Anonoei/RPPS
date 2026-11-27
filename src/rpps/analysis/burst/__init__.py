"""TDMA bursts demodulated one by one: each acquired on the preamble the bursts share, as a TDMA receiver must

    r = rp.analysis.burst.demodulate(x, Fs)                       # or mod=, baud=, preamble= (symbols) when known
    print(r)       # "12 bursts, qpsk at 100.0 kBd, a 48-symbol preamble; frame 4.615 ms, 8 slots; EVM 6.1% median"
    b = r.bursts[0]
    b.symbols, b.bits, b.cfo, b.evm, b.slot, b.frame   # carrier-, timing- and phase-corrected symbols, decided bits

Every burst comes from another transmitter (or the same one after a gap), with its own carrier offset, phase and symbol
timing, and is too short to train loops on: so each is acquired on its own, data-aided, from the preamble.

  1. rpps.analysis.framing.frames.bursts gates the bursts (bursts that touch cut apart at a guard time or a power step:
     frames.split) and acquires each blind (rate, modulation, offset); bursts too short
     for that keep only their span. The modulation is the bursts' majority, the rate their median (or `mod` / `baud`).
  2. The preamble: the leading run of symbols the bursts agree on, as phase steps (rpps.analysis.framing.frames.preamble, or
     `shared_preamble` for a short one, whichever is longer), or `preamble` given. A network's burst types each carry a
     unique word of their own: when the bursts share fewer than `min_shared` symbols, they are grouped by length and
     each group's own preamble found (`burst_kinds`); each burst is acquired on its kind's. Known symbols elsewhere (a
     mid-amble, a postamble: `shared_segments`) serve the same way, at their offset; with none at all, each burst is
     returned as acquired blind (`Bursts.blind`: its phase known only to the constellation's symmetry). Its symbols are rebuilt from the steps and turned onto the constellation (one rotation for all
     bursts: the M-fold phase ambiguity is the same for every burst, so their bits agree with each other).
  3. In each burst the preamble is found by correlating its shaped waveform in pieces (the pieces' magnitudes added: a
     carrier offset costs little), then the fractional timing that best matches the preamble's phase steps (blind to
     the carrier), then from z_k = y_k conj(p_k) over the preamble: the carrier offset (the mean turn of z_k z*_{k-1})
     and the phase and gain (sum z_k with the turn taken out).
  4. The burst's symbols, from the preamble to the end of the gate, are taken at that timing and corrected. A burst of
     several `window`s has its symbol clock followed (`retime`): the matched filter's energy peak in each window
     (rpps.sync.time.mat.best_tau), a line through them anchored on the preamble (the clock offset, `clock`). A second-
     phase from decisions on the constellation's points, tracked across 32-symbol windows (rpps.sync.phase.decided), follows what is left of the carrier
     through the data; decisions give bits, the soft demapper LLRs (`llr`, noise from the burst's own symbols), and the error vector magnitude.
  5. The TDMA structure: the period of the bursts' starts (rpps.analysis.spectrum.waveform.on_off, else `frame`) and the slots,
     clusters of the starts within a frame (to within `slot_tol` symbols): each burst's frame and slot.
Limits: the clock is followed as a straight line (a constant offset over the burst), from windows of a few hundred
symbols; a burst whose preamble is weaker than the data's noise floor allows is reported with `ok` False and no bits.

A network's burst time plan (rpps.plan) from the bursts rp.sync.Bursts detects: `learn_plan` (the BTU, the frame, each
carrier's slots and the kinds of burst in them), and whether a plan, learned or told, explains the bursts:
`check_plan` (each burst in a slot of its carrier and kind, the plan's origin and clock fitted to them; rp.viz.bursts
draws it, folded on the superframe or along the capture).

Modules: `preamble` (what the bursts share), `demod` (each burst acquired and demodulated; the frame and slots),
`plan` (units, learn_plan, check_plan).

Burst code elsewhere: what a receiver runs is in the data path, which may not import this package.
  - rpps.sync.burst: `Burst` (bursts gated from a stream by energy), `BurstTx` (bursts placed in a stream)
  - rpps.sync.plan: `Bursts` (bursts cut where a burst time plan puts them), `PlanTx`
  - rpps.profile.burst: the runners for a profile's `burst` step (receive and transmit)
  - rpps.plan: burst time plans as definitions; rpps.viz.plots.bursts draws bursts against them
"""
from . import preamble, demod, plan
from .preamble import shared_preamble, preamble_symbols, Segment, shared_segments, periodic_word, burst_kinds
from .demod import BurstResult, Bursts, acquire, retime, demodulate
from .plan import Units, units, NoPlan, learn_plan, PlanCheck, check_plan
