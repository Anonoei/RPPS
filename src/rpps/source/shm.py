"""Channels in shared memory: one process channelizes, others read their channel (on pysdrlib's ring)

    wide = rp.source.Device("hackrf", cf=915e6, Fs=20e6)(65536)
    wide + rp.signal.filters.Channelizer(8) >> rp.source.ShmSink([f"ch{k}" for k in range(8)])   # one process
    for blk in rp.source.Shm("ch3")(8192) + rx: ...                                                # any other

A process that runs several demodulators does the wideband work once and publishes each channel; each demodulator is a
process of its own, reading at its own pace (the GIL is not shared, a crash is its own). Several readers can read one
channel: a live view (pyspecan) and a decoder alike.

The ring is pysdrlib's (`pysdrlib.remote.shm`: complex64, one writer, a cursor per reader, absolute sample indices with gaps).
Samples are published at their `meta.pos`, so a gap upstream (an overrun, a dropped block) is a gap in the ring, and a
reader's positions stay true to time; a reader that falls a ring behind is moved on and counts what it lost in
`stream.dropped`.

What a ring does not carry is in a manifest beside it, `<dir>/<name>.json` (dir: `RPPS_SHM_DIR`, else `rpps-shm` in the
system's temp directory), written whole (write, then rename), and listed in `<dir>/index.json` ({name: {manifest, gen}}):

    segment     the ring's shared-memory name (pysdrlib's ShmReader opens it)
    gen         bumped when the channel's rate or frequency changes: a new ring, the old one closed (readers start over)
    Fs, cf, sps the channel's rate and centre; bw its usable width (the sink's `bw`, else Fs)
    parent      {"Fs", "cf"} of the wideband stream it was cut from (a channelizer's), or None
    t0          the time of the channel's index 0 on the source's clock (`stream.t0`), or None
    wall_time, wall_index   time.time() when the first block was published, and that block's first index (an epoch anchor)
    label, profile          what a view shows: free text, the profile decoding it (or None)
    dtype       "cf32"
"""
import json
import os
import re
import tempfile
import time

import numpy as np

from .. import base
from .source import Source

_NAME = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")

def shm_dir(directory=None):
    """Where the manifests are: `directory`, `RPPS_SHM_DIR`, else `rpps-shm` in the system's temp directory"""
    return directory or os.environ.get("RPPS_SHM_DIR") or os.path.join(tempfile.gettempdir(), "rpps-shm")

def _write_json(path, obj):
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)

def _locked(directory):
    """A lock on the directory's index (posix; elsewhere the index is written unlocked)"""
    class _Lock:
        def __enter__(self):
            self.f = open(os.path.join(directory, ".lock"), "a")
            try:
                import fcntl
                fcntl.flock(self.f, fcntl.LOCK_EX)
            except ImportError:
                pass
            return self
        def __exit__(self, *a):
            self.f.close() # (closing releases the lock)
    return _Lock()

def read_index(directory=None):
    """{name: {"manifest": path, "gen": n}} of the channels published in `directory` (what a picker lists)"""
    try:
        with open(os.path.join(shm_dir(directory), "index.json")) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def _index_set(directory, name, entry):
    with _locked(directory):
        idx = read_index(directory)
        if entry is None:
            idx.pop(name, None)
        else:
            idx[name] = entry
        _write_json(os.path.join(directory, "index.json"), idx)

def manifest(name, directory=None):
    """The manifest of channel `name`, or None when nothing publishes it"""
    try:
        with open(os.path.join(shm_dir(directory), f"{name}.json")) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None

def _check(name):
    if not _NAME.match(str(name)):
        raise ValueError(f"channel name {name!r}: letters, digits, '_', '-', '.' (up to 64)")
    return str(name)

class _Channel:
    """One ring and its manifest"""
    def __init__(self, sink, name):
        self.sink, self.name = sink, name
        self.writer, self.gen, self.key, self.head = None, -1, None, None

    def publish(self, x, pos, info):
        from pysdrlib.remote.shm import ShmWriter
        key = (info["Fs"], info["cf"])
        if self.writer is None or key != self.key or (self.head is not None and pos < self.head):
            self._close_ring()
            self.gen += 1
            self.writer = ShmWriter(self.sink.capacity, stream_id=self.gen)
            self.key, self.head = key, None
            m = dict(info, segment=self.writer.name, gen=self.gen, dtype="cf32", wall_time=time.time(), wall_index=int(pos),
                     label=self.sink.labels.get(self.name), profile=self.sink.profile)
            path = os.path.join(self.sink.dir, f"{self.name}.json")
            _write_json(path, m)
            _index_set(self.sink.dir, self.name, {"manifest": path, "gen": self.gen})
        if len(x):
            self.writer.publish(np.ascontiguousarray(x, dtype=np.complex64), int(pos))
            self.head = int(pos) + len(x)

    def _close_ring(self):
        if self.writer is not None:
            self.writer.destroy() # (readers that have it keep it mapped: they drain it, see it closed, read the manifest again)
            self.writer = None

    def close(self):
        self._close_ring()
        for path in (os.path.join(self.sink.dir, f"{self.name}.json"),):
            try:
                os.remove(path)
            except FileNotFoundError:
                pass
        _index_set(self.sink.dir, self.name, None)

class ShmSink:
    """Publish blocks to shared memory, for other processes: `block >> sink`, `blocks >> sink`

    name: the channel's name, or one per column of a channelized block (`Channelizer` output, (time, M)): column k is the
    channel centred on cf + k Fs (above half the channels, the negative frequencies), at the block's rate.
    capacity: samples a ring holds (pysdrlib's `fit_capacity` keeps it to what shared memory has room for); a reader more
    than that behind is moved on. labels: {name: text} for a view; profile: the profile that decodes these, if any.
    bw: the usable width of a channel [Hz] (a view boxes it inside the span); None: the channel's rate. A critically
    sampled channelizer's channels overlap at their edges, so its flat passband is a little under the rate.
    directory: where the manifests go (`shm_dir`). Close it (or use `with`) to remove the rings and manifests."""
    def __init__(self, name, capacity=1 << 20, labels=None, profile=None, directory=None, bw=None):
        from pysdrlib.remote.shm import fit_capacity
        self.names = [_check(n) for n in ([name] if isinstance(name, str) else name)]
        self.capacity = fit_capacity(capacity)
        if not self.capacity:
            raise MemoryError("shared memory has no room for a ring")
        self.labels, self.profile, self.bw = dict(labels or {}), profile, bw
        self.dir = shm_dir(directory)
        os.makedirs(self.dir, exist_ok=True)
        self._ch = {n: _Channel(self, n) for n in self.names}
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
        t0 = m.stream.t0
        for k, n in enumerate(self.names):
            off = ((k + M//2) % M - M//2)*Fs if x.ndim > 1 else 0.0
            info = {"Fs": Fs, "cf": cf + off, "sps": float(m.sps or 0.0), "bw": float(self.bw) if self.bw else Fs, "parent": parent,
                    "t0": None if t0 is None else float(t0)}
            self._ch[n].publish(cols[:, k], m.pos, info)
        self.count += len(x)
        return self

    def close(self):
        for ch in self._ch.values():
            ch.close()

    def __enter__(self):
        return self
    def __exit__(self, *a):
        self.close()

class Shm(Source):
    """A channel another process publishes (`ShmSink`): blocks at its rate and centre, positions counted from where this
    reader started, gaps and what it missed in `meta.pos` and `stream.dropped`

    backlog: samples to start back from the newest (0: from now). poll: seconds to wait when nothing is new (the ring has
    no wakeup). timeout: seconds with nothing new, or no channel, before the blocks end (None: wait for ever).
    When the publisher changes the channel (a new `gen`: a retune), the stream starts over (`restart`) at the new rate.
    When it stops, the blocks end."""
    def __init__(self, name, backlog=0, poll=0.002, timeout=None, directory=None):
        self.name, self.backlog, self.poll, self.timeout, self.dir = _check(name), int(backlog), poll, timeout, directory
        m = self._wait_manifest()
        if m is None:
            raise FileNotFoundError(f"no channel {name!r} in {shm_dir(directory)}")
        super().__init__(m["Fs"], m["cf"], m.get("sps") or 0.0)
        self.m, self._reader = m, None

    def __str__(self):
        return f"Shm({self.name}: {self.Fs/1e3:g} kS/s at {self.cf/1e6:g} MHz)"

    def _wait_manifest(self, newer=-1):
        t = time.monotonic()
        while True:
            m = manifest(self.name, self.dir)
            if m is not None and m["gen"] > newer:
                return m
            if self.timeout is not None and time.monotonic() - t > self.timeout:
                return None
            time.sleep(max(self.poll, 0.01))

    def _open(self, m):
        from pysdrlib.remote.shm import ShmReader
        self._close_reader()
        self.m = m
        self._reader = ShmReader(m["segment"])
        self.Fs, self.cf, self.sps = m["Fs"], m["cf"], m.get("sps") or 0.0
        start = max(0, self._reader.head - self.backlog)
        return start

    def _close_reader(self):
        if self._reader is not None:
            self._reader.close()
            self._reader = None

    def close(self):
        self._close_reader()

    def _samples(self, count):
        s = self.stream
        try:
            pos = self._open(self.m)
            base_ = pos - s.pos # ring index of this stream's item 0
            if s.t0 is None and self.m.get("t0") is not None:
                s.t0 = self.m["t0"] + base_/self.Fs
            buf = np.empty(int(count), np.complex64)
            have, idle, started = 0, time.monotonic(), False
            while True:
                n, first = self._reader.read(buf[have:], pos)
                if first > pos and not started: # (a backlog reaching before the ring's first sample: start where it is)
                    base_ += first - pos
                    pos = first
                    if self.m.get("t0") is not None:
                        s.t0 = self.m["t0"] + base_/self.Fs
                elif first > pos: # lapped, or a gap: what follows starts further on
                    if have:
                        yield self._block(buf[:have].copy())
                        have = 0
                    s.dropped += first - pos
                    s.pos = first - base_
                    pos = first
                if n:
                    started = True
                    have += n
                    pos += n
                    idle = time.monotonic()
                    if have == len(buf):
                        yield self._block(buf.copy())
                        have = 0
                    continue
                if self._reader.closed and self._reader.head <= pos: # The publisher stopped, or changed the channel
                    m = self._wait_manifest(self.m["gen"]) if self.timeout != 0 else None
                    if m is None:
                        yield self._block(buf[:have].copy(), flush=True)
                        return
                    if have:
                        yield self._block(buf[:have].copy(), flush=True)
                        have = 0
                    self.restart()
                    pos = self._open(m)
                    base_, started = pos, False
                    s = self.stream
                    s.src_Fs = self.Fs
                    if m.get("t0") is not None:
                        s.t0 = m["t0"] + base_/self.Fs
                    continue
                if have: # Give what is here rather than wait for a whole block: a reader stays live
                    yield self._block(buf[:have].copy())
                    have = 0
                if self.timeout is not None and time.monotonic() - idle > self.timeout:
                    yield self._block(buf[:0].copy(), flush=True)
                    return
                time.sleep(self.poll)
        finally:
            self._close_reader()
