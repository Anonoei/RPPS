"""Building on RPPS: a lock that follows a signal, and a profile's messages from a selected band

    python examples/integrating.py

Two parts of the stable API (docs/guide/integrating.md), as a display or a monitor uses them:

- `rp.sync.Lock`: a signal's symbols as a snapshot. `Lock.find` searches once (rate, modulation, roll-off, carrier
  offset); `lock.update(block)` follows the same signal on any later block, consecutive or not, with no search, and
  `lock.stale` says why it is time to search again. Nothing is told the signal.
- `rp.profile.Decoder`: a band picked out of a capture (its offset from the centre, at the capture's own rate) to the
  profile's messages. It shifts, decimates and resamples to the rate the profile needs (`rp.profile.info(...).requires`),
  takes blocks with their sample index and time, and gives each frame as an `rp.profile.Message` with its CRC verdict,
  where it was in the caller's samples, and when.
"""
import numpy as np

import rpps as rp

def lock(rng):
    Fs, baud = 1e6, 125e3
    bits = rng.integers(0, 2, 2*40000).astype(bool)
    syms = rp.Data(bits, Fs=baud, sps=1) * rp.symbol.load("psk/qpsk")
    air = (syms + rp.wave.Shape(Fs/baud, rp.signal.filters.load("pulse/rrc/0p35"))
           + rp.signal.channel.CFO(1.2e3) + rp.signal.channel.AWGN(esn0_db=18, power=None, seed=3))

    lk = rp.sync.Lock.find(air.data[:20000], Fs)
    print(f"found:     {lk.det.mod} at {lk.det.baud:.0f} Bd, cfo {lk.det.cfo:+.0f} Hz, Es/N0 {lk.quality.esn0_db:.1f} dB")
    later = lk.update(air.data[150000:160000]) # far from the first block: no search, the same signal
    print(f"followed:  {len(later.symbols)} symbols, Es/N0 {later.quality.esn0_db:.1f} dB, stale: {later.stale}")
    noise = (rng.standard_normal(10000) + 1j*rng.standard_normal(10000))/np.sqrt(2)
    gone = lk.update(noise)
    print(f"signal off: stale because {gone.stale}")

    assert lk.det.mod == "qpsk" and abs(lk.det.baud/baud - 1) < 1e-3 and abs(lk.det.cfo - 1.2e3) < 50
    assert later.stale is None and later.det.mod == "qpsk" and abs(later.quality.esn0_db - 18) < 1.5
    assert gone.stale

def decoder(rng):
    # Three ADS-B squitters from the profile's own transmit side, at its 2 MS/s
    p = rp.profile.load("adsb/mode_s")
    tx, stream, parts, payloads = p.tx(), None, [np.zeros(4000, np.complex64)], []
    for i in range(3):
        bits = rng.integers(0, 2, 83).astype(bool)
        d = rp.Data(bits, Fs=2e6, sps=1, stream=stream, flush=i == 2)
        stream = d.stream
        parts += [tx(d, fields={"df": 17}).data*0.5, np.zeros(3000, np.complex64)]
        payloads.append(np.packbits(bits).tobytes())

    # As a 2.4 MS/s radio tuned 150 kHz below 1090 MHz sees them
    Fs, offset = 2.4e6, 150e3
    x = rp.signal.sample.fractional_resample(np.concatenate(parts), Fs/2e6, 2e6)
    capture = rp.Data(x.astype(np.complex64), Fs=Fs) + rp.signal.channel.CFO(offset) + rp.signal.channel.AWGN(snr_db=25, seed=4)
    print(f"adsb/mode_s needs {rp.profile.info('adsb/mode_s').requires}")

    dec = rp.profile.Decoder("adsb/mode_s", Fs=Fs, offset=offset, cf=1090e6 - offset)
    msgs, t0 = [], 1700000000.0 # the capture's start, as a radio stamps it
    for a in range(0, len(capture), 8192): # blocks as they come, each with where it starts
        msgs += dec.feed(capture.data[a:a + 8192], start=a, time=t0 + a/Fs)
    msgs += dec.close()
    # Failed frames come too (crc_ok False): here a preamble found in the noise before the first squitter
    for m in msgs:
        what = m.payload.hex() if m.crc_ok else "crc failed"
        print(f"{m.kind} at sample {m.sample} (+{(m.time - t0)*1e3:.3f} ms) on {m.freq/1e6:g} MHz: {what}")
    print(f"decoder: {dec.status['state']}, {dec.status['good']} good")

    good = [m for m in msgs if m.crc_ok]
    assert [m.payload for m in good] == payloads
    assert all(m.freq == 1090e6 and abs(m.time - (t0 + m.sample/Fs)) < 1e-6 for m in good) # placed in the caller's samples
    assert dec.status["state"] == "locked"

def main():
    rng = np.random.default_rng(5)
    lock(rng)
    decoder(rng)

if __name__ == "__main__":
    main()
