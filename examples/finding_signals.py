"""Boxes on the signals in a capture, at once or block by block for a display

    python examples/finding_signals.py

`rp.analysis.find_signals(x, Fs)` gives each signal's box: its edges (`lo`, `hi`, from the centre frequency), its SNR
over what it stands on, and when it is on (`t0`, `t1`, and `bursts` for one that comes and goes). The false-alarm rate
is the setting (`pfa`: expected false boxes in a capture of pure noise), not a threshold in dB. A narrower signal on a
wider one (a jammer on a carrier, carriers on a transponder's pedestal) is a box of its own, `over` the wider one,
which `holds` it.

`rp.analysis.spectrum.find.Finder` does the same on a stream, one block at a time, for a display: ids stay with
their signals from block to block, and what it finds does not depend on how the stream is cut into blocks. Effort
levels choose how hard it looks (`rp.effort("fast")` for a display, a few milliseconds a block).

Here: three RRC carriers at different levels, a narrow one jamming the widest, and a TDMA burst train.
"""
import numpy as np

import rpps as rp
from rpps.analysis.spectrum.find import Finder

Fs, N = 10e6, 1 << 20

def carrier(rng, baud, f, snr_db):
    """QPSK at `baud`, RRC 0.35, `f` from the centre, `snr_db` over the noise in its band"""
    sps = int(Fs//baud)
    d = rp.Data(rng.integers(0, 2, 2*(N//sps + 1)).astype(bool), Fs=baud, sps=1) * rp.symbol.load("psk/qpsk")
    d = d + rp.wave.Shape(sps, rp.signal.filters.load("pulse/rrc/0p35"))
    x = np.asarray(d.data)[:N]
    x = x/np.sqrt(np.mean(np.abs(x)**2))*np.sqrt(10**(snr_db/10)*1.35*baud/Fs)
    return x*np.exp(2j*np.pi*f/Fs*np.arange(N))

def capture():
    rng = np.random.default_rng(1)
    x = (rng.standard_normal(N) + 1j*rng.standard_normal(N))/np.sqrt(2)
    x = x + carrier(rng, 1e6, -2.5e6, 15) + carrier(rng, 500e3, 0.0, 10) + carrier(rng, 1.25e6, 2.6e6, 20)
    x = x + carrier(rng, 100e3, 2.8e6, 30)                     # jamming the 1.25 MBd carrier
    gate = (np.arange(N)//(N//8)) % 2 == 1                     # on half the time: four bursts
    x = x + carrier(rng, 250e3, -4e6, 15)*gate
    return x.astype(np.complex64)

def main():
    x = capture()
    sv = rp.analysis.find_signals(x, Fs, cf=2.4e9)
    for s in sv:
        print(f"  {s.id}: {s.describe()}")
    wide = {round(s.mid/1e5)/10: s for s in sv}                # (by centre, MHz to 0.1)
    assert set(wide) == {-4.0, -2.5, 0.0, 2.6, 2.8}, set(wide)
    jammer, jammed = wide[2.8], wide[2.6]
    assert jammer.over is jammed and jammed.holds == 1
    assert len(wide[-4.0].bursts) == 4                         # the burst train: a time box per burst
    assert abs(wide[-2.5].width/1e6 - 1.17) < 0.1              # an RRC carrier's box: about 1.17 x its baud

    # Block by block, as a display would: the same boxes once it holds enough frames, ids kept
    with rp.effort("fast"):
        f = Finder(Fs, nfft=4096, cf=2.4e9)
        ids = []
        for i in range(0, N, 65536):
            live = f.step(x[i:i + 65536])
            if live is not None:
                ids.append({round(s.mid/1e5)/10: s.id for s in live if s.mid > -3.5e6})
    print(f"\nlast block: {len(live)} signals; {', '.join(f'{m:+.1f} MHz' for m in sorted(ids[-1]))} (steady ones)")
    assert set(ids[-1]) == {-2.5, 0.0, 2.6, 2.8}, ids[-1]
    assert all(d == ids[-1] for d in ids[-4:])                 # a steady signal keeps its id

if __name__ == "__main__":
    main()
