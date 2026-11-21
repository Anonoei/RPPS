"""What is in a capture: find the signals, then work out the steps that process each one

    python examples/unknown_signal.py

`rp.analysis` stands outside the data path. `survey(source)` finds the signals in a capture: where, how wide, how
strong, roughly what symbol rate. `profile(source)` then takes each one through the receive chain step by step
(channel, acquisition, timing, carrier, the bit map, codes, framing, scrambling), trying the shipped definitions at
each step. A signal whose every step is known names the profile that receives it; where nothing fits, the profile ends
with an UNSOLVED step and what it might be. `blind=True` also runs the blind tools there (codes, scramblers and frames
that are in no definition, as a `Guess` you can save as one), which takes tens of seconds a signal.

For a live radio, examples/live_profile.py does the same on a capture taken into memory first (the analysis
runs slower than the radio).

Here: an 8 MS/s capture holding a DVB-S carrier (the shipped dvbs/qpsk_3_4 transmitter: convolutional code 3/4,
Reed-Solomon, interleaver, energy dispersal) and an uncoded BPSK carrier, which no definition can explain.
"""
import time

import numpy as np

import rpps as rp

Fs = 8e6

def capture():
    rng = np.random.default_rng(0)
    # DVB-S: MPEG transport stream packets (0x47 sync byte, 187 bytes) at 1 MBd, 8 samples a symbol, 1.7 MHz up
    ts = np.zeros((900, 188), np.uint8)
    ts[:, 0] = 0x47
    ts[:, 1:] = rng.integers(0, 256, (900, 187))
    dvbs = rp.profile.load("dvbs/qpsk_3_4", sps=8).tx()(rp.Data(np.unpackbits(ts.reshape(-1)).astype(bool), Fs=1e6, sps=1,
                                                                flush=True))
    n = len(dvbs)
    # BPSK at 250 kBd, 2 MHz down
    bpsk = rp.Data(rng.integers(0, 2, n//32 + 1).astype(bool), Fs=Fs/32, sps=1) * rp.symbol.load("psk/bpsk")
    bpsk = bpsk + rp.wave.Shape(32, rp.signal.filters.load("pulse/rrc/0p35"))
    a = (rp.Data(dvbs.data, Fs=Fs) + rp.signal.channel.CFO(1.7e6)).data
    b = (rp.Data(bpsk.data[:n], Fs=Fs) + rp.signal.channel.CFO(-2.0e6)).data
    x = a/np.sqrt(np.mean(np.abs(a)**2)) + 0.5*b/np.sqrt(np.mean(np.abs(b)**2))
    x = x + 0.1*(rng.standard_normal(n) + 1j*rng.standard_normal(n))
    return rp.source.Memory(x.astype(np.complex64), Fs=Fs, cf=1.2e9)

def main():
    mem = capture()
    print(f"{mem}\n")

    survey = rp.analysis.survey(mem)
    for i, s in enumerate(survey):
        print(f"  {i}: {s.describe()}")
    assert len(survey) == 2

    t = time.perf_counter()
    profiles = rp.analysis.profile(mem, survey=survey, blind=False)
    for p in profiles:
        print(f"\n{p}")
    print(f"\n({time.perf_counter() - t:.0f} s)")

    named = {p.profile_name for p in profiles if p.solved}
    assert named == {"dvbs/qpsk_3_4"}, named
    assert sum(not p.solved for p in profiles) == 1 # the BPSK carrier: no code fits

    # The profile it named receives the signal from here on: p.rx() (or `python -m rpps rx dvbs/qpsk_3_4 ...`)
    dvbs = next(p for p in profiles if p.solved)
    print(f"receive it with: {dvbs.rx()}")

if __name__ == "__main__":
    main()
