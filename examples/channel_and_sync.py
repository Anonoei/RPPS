"""Test a receiver against a simulated channel, and recover the signal with block estimators

    python examples/channel_and_sync.py

The channel is a chain of stages like any other (`rp.signal.channel`: AWGN, CFO, PhaseOffset, ClockDrift, Multipath,
Fading, Gain), so the impairments are the ones a radio gives and they stream block by block.

The receiver uses the first-class sync of rpps.sync: estimate from a window with block operations,
then carry the estimate across windows with a slow loop. `Acquire` finds the symbol rate, modulation, roll-off and
frequency offset from the stream itself; `time.Mat` takes the symbols with a matrix matched filter at the detected rate;
`phase.FFCarrier` removes frequency and phase feed-forward; `quality.Monitor` reports Es/N0 as the symbols go by.
Nothing is told the baud rate or the offsets.

The same estimators receive a real capture in examples/e2e.py (Oerder & Meyr timing, the M-th power carrier) and
e2e_mat.py (the matrix matched filter).
"""
import numpy as np

import rpps as rp

def main(esn0_db=15.0, cfo=3.1e3, ppm=20.0):
    rng = np.random.default_rng(1)
    Fs, baud = 1e6, 125e3

    # Transmit: QPSK, root-raised-cosine pulses at 8 samples a symbol
    qpsk = rp.symbol.load("psk/qpsk")
    syms = rp.Data(rng.integers(0, 2, 2*30000).astype(bool), Fs=baud, sps=1) * qpsk
    tx = syms + rp.wave.Shape(Fs/baud, rp.signal.filters.load("pulse/rrc/0p35"))

    # The channel: a frequency offset, a phase, a sample clock 20 ppm fast, a weak echo, and noise at Es/N0 (power=None
    # measures the signal: the pulse shaping changed its power)
    channel = (rp.signal.channel.CFO(cfo) + rp.signal.channel.PhaseOffset(0.7) + rp.signal.channel.ClockDrift(ppm)
               + rp.signal.channel.Multipath([1, 0.1j], [0, 3]) + rp.signal.channel.AWGN(esn0_db=esn0_db, power=None, seed=2))
    air = tx + channel

    # Receive: the samples as a radio would give them, in blocks; nothing about the signal is given
    rx = rp.sync.Acquire() + rp.sync.time.Mat() + rp.sync.phase.FFCarrier() + rp.sync.quality.Monitor()
    n = 0
    for out in rp.source.Memory(air.data, Fs=Fs)(16384) + rx:
        n += len(out)
    det, q = out.det, out.stream.est["quality"]
    print(f"{n} symbols out of {len(syms)} sent")
    print(f"detected: {det}")
    print(f"measured: {q}")

    assert det.mod == "qpsk"
    assert abs(det.baud/baud - 1 - ppm*1e-6) < 20e-6 # the clock error shows up as the symbol rate
    assert abs(det.cfo - cfo) < 20
    assert abs(q.esn0_db - esn0_db) < 1.0 # within 1 dB of the channel: a bigger gap is a diagnostic, not noise

if __name__ == "__main__":
    main()
