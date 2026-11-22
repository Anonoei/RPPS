"""Several signals at once: one process channelizes, the others each receive a channel from shared memory

    python examples/channel_shm.py

The wideband work (here a polyphase channelizer) is done once, and each channel is published as a pysdrlib shared-memory
stream (`rp.source.ShmSink`). A demodulator receives its channel like a radio, `rp.source.Device("shm@rpps.ch1")`, in a
process of its own (`python my_rx.py ch1`); here threads stand in for those processes. Positions are true to time, a
retune starts a receiver over, the publisher stopping ends its blocks, and a live view (pyspecan) or `pysdrlib.find()`
sees the same channels.

Here: two tones in a 400 kS/s capture, channelized into 4 channels of 100 kS/s; each receiver finds its tone and says
where it is in absolute frequency, from the channel's own centre. `stream.realtime()` says how much of real time a
receiver used: under 1.0 it keeps up.
"""
import os
import tempfile
import threading
import time

import numpy as np

import rpps as rp

def main():
    os.environ.setdefault("PYSDRLIB_SHM_DIR", tempfile.mkdtemp(prefix="rpps-shm-")) # (a directory of its own, for the example)
    Fs, cf, M = 400e3, 100e6, 4
    n = 200_000
    t = np.arange(n)/Fs
    tones = {"ch1": 100e3 + 7e3, "ch3": -100e3 - 12e3} # inside channels 1 and 3 (channel 3 is -100 kHz)
    x = sum(np.exp(2j*np.pi*f*t) for f in tones.values()).astype(np.complex64)
    names = [f"ch{k}" for k in range(M)]

    found = {}
    def receiver(name):
        src = rp.source.Device(f"shm@rpps.{name}")
        y = np.concatenate([np.asarray(b.data) for b in src(4000) + rp.signal.DCBlock()]) # (a demodulator would go here)
        f = np.fft.fftfreq(len(y), 1/src.Fs)[np.argmax(np.abs(np.fft.fft(y)))]
        found[name] = (src.cf + f, src.stream.realtime())

    wide = iter(rp.source.Memory(x, Fs=Fs, cf=cf)(20_000) + rp.signal.filters.Channelizer(M))
    with rp.source.ShmSink(names, labels={"ch1": "tone A", "ch3": "tone B"}) as sink:
        next(wide) >> sink # The channels exist from the first block: receivers can open them
        threads = [threading.Thread(target=receiver, args=(nm,)) for nm in tones]
        for th in threads:
            th.start()
        time.sleep(0.3) # (a receiver gets what is published once it is receiving)
        for blk in wide: # The channelizer process: wideband blocks in, one stream per channel out
            blk >> sink
    for th in threads: # Closing the sink ended the streams, and with them the receivers' blocks
        th.join()

    for name, f in tones.items():
        got, load = found[name]
        print(f"{name}: tone at {got/1e6:.4f} MHz (sent {(cf + f)/1e6:.4f}), load {load:.3f}")
        assert abs(got - (cf + f)) < 100, (name, got, cf + f)

if __name__ == "__main__":
    main()
