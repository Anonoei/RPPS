"""Channels in shared memory: one process channelizes, others receive their channel (pysdrlib's `shm` streams)

    wide = rp.source.Device("hackrf", cf=915e6, Fs=20e6)(65536)
    wide + rp.signal.filters.Channelizer(8) >> rp.source.ShmSink([f"ch{k}" for k in range(8)])   # one process
    for blk in rp.source.Device("shm@rpps.ch3")(8192) + rx: ...                                     # any other

A process that runs several demodulators does the wideband work once and publishes each channel; each demodulator is a
process of its own, receiving at its own pace (the GIL is not shared, a crash is its own). Any number of receivers can
take one channel: a decoder and a live view (pyspecan) alike, and `pysdrlib.find()` lists them.

The streams, their rings and manifests are pysdrlib's (`pysdrlib.shm.ShmStream`, which owns the format): a channel is
received as a pysdrlib device, `rp.source.Device("shm@<namespace>.<name>")`, with positions true to time (a gap upstream
is a gap there, a receiver that falls behind counts what it missed in `stream.dropped`), a retune starting the stream
over, and the end of the publisher ending the blocks.
"""
import numpy as np

from .. import base

class ShmSink:
    """Publish blocks to shared memory, for other processes: `block >> sink`, `blocks >> sink`

    name: the channel's name, or one per column of a channelized block (`Channelizer` output, (time, M)): column k is the
    channel centred on cf + k Fs (above half the channels, the negative frequencies), at the block's rate. Each is
    published as `<namespace>.<name>` (namespace None: the bare name). capacity: samples a ring holds (None: pysdrlib's
    default, a second); a receiver more than that behind is moved on. labels: {name: text} for a view; profile: the
    profile that decodes these, if any; bw: a channel's usable width [Hz] (None: its rate; a critically sampled
    channelizer's flat passband is a little under it). directory: where the manifests go (pysdrlib's default:
    `$PYSDRLIB_SHM_DIR`). Close it (or use `with`) to end the streams.

    A block's `meta.pos` is where it goes in its stream, so a gap upstream stays a gap. A change of rate or centre, or a
    stream that starts over, is a retune: a new generation, which receivers follow."""
    def __init__(self, name, namespace="rpps", capacity=None, labels=None, profile=None, bw=None, directory=None):
        self.names = [name] if isinstance(name, str) else list(name)
        self.full = {n: f"{namespace}.{n}" if namespace else n for n in self.names}
        self.capacity, self.labels, self.profile, self.bw, self.dir = capacity, dict(labels or {}), profile, bw, directory
        self._out = {} # name -> [ShmStream, (Fs, cf), pos of the generation's first sample, the Stream's gen]
        self.count = 0

    def write(self, data):
        if not isinstance(data, base.data._Data):
            raise TypeError("ShmSink publishes blocks (rp.Data): their rate, frequency and position go with them")
        x, m = np.asarray(data.data), data.meta
        cols = x.reshape(len(x), -1) if x.ndim > 1 else x[:, None]
        if cols.shape[1] != len(self.names):
            raise ValueError(f"{cols.shape[1]} channel(s) in the block, {len(self.names)} name(s)")
        M, Fs, cf = cols.shape[1], float(m.Fs), float(m.cf)
        parent = {"Fs": Fs*M, "cf": cf} if x.ndim > 1 else None
        for k, n in enumerate(self.names):
            ccf = cf + (((k + M//2) % M - M//2)*Fs if x.ndim > 1 else 0.0)
            self._publish(n, cols[:, k], m, Fs, ccf, parent)
        self.count += len(x)
        return self

    def _info(self, n, m, parent):
        """What a view shows of channel n (None: left out)"""
        return {"bw": float(self.bw) if self.bw else float(m.Fs), "sps": float(m.sps or 0.0), "parent": parent,
                "t0": None if m.stream.t0 is None else float(m.stream.t0), "label": self.labels.get(n),
                "profile": self.profile}

    def _publish(self, n, x, m, Fs, cf, parent):
        from pysdrlib.shm import ShmStream
        cur = self._out.get(n)
        if cur is None:
            info = {k: v for k, v in self._info(n, m, parent).items() if v is not None}
            out = ShmStream(self.full[n], Fs=Fs, cf=cf, capacity=self.capacity, directory=self.dir, **info)
            cur = self._out[n] = [out, (Fs, cf), m.pos, m.stream.gen]
        elif cur[1] != (Fs, cf) or m.pos < cur[0].head + cur[2] or m.stream.gen != cur[3]:
            # A retune, or a stream that started over: a new generation, its indices from its first sample
            cur[0].retune(Fs=Fs, cf=cf, **self._info(n, m, parent))
            cur[1:] = [(Fs, cf), m.pos, m.stream.gen]
        if len(x):
            cur[0].publish(np.ascontiguousarray(x, dtype=np.complex64), int(m.pos - cur[2]))

    def close(self):
        for out, *_ in self._out.values():
            out.close()
        self._out = {}

    def __enter__(self):
        return self
    def __exit__(self, *a):
        self.close()
