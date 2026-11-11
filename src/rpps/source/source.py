"""Sources and sinks of signal blocks

    src = rp.source.File("ci32t", "capture.cs16", Fs=2e6, cf=100e6)
    for out in src(4096) + (lpf + dec):          # blocks flow through the pipeline, its state carried along
        ...
    src(4096) + (lpf + dec) >> rp.source.Sink("cf64", "out.iq")   # or straight into a sink

Threads: `.threaded()` puts everything before it in its own thread, and `.pipelined(pipe)` gives every stage of a
pipeline its own, connected by bounded queues. Stages overlap where they release the GIL (numpy, the rpps._native
kernels), and a radio is drained while the stages work:

    src(16384).threaded(drop=True).pipelined(agc + costas + mm) >> sink
"""
import queue
import threading

import numpy as np
import pysdrlib

from .. import base
from ..base import Stream, Data
from ..errors import StreamError
from . import formats as fmts

def _close(it):
    """Close a generator: its `finally` runs now (a radio stops, a thread exits)"""
    close = getattr(it, "close", None)
    if close is not None:
        close()

class _End:
    """End of a threaded stream: carries the producer's exception, if any"""
    __slots__ = ("exc",)
    def __init__(self, exc=None):
        self.exc = exc

def _threaded(gen, depth, drop, name):
    """Iterate `gen` in a thread of its own, handing its items over through a queue of `depth`"""
    q = queue.Queue(maxsize=max(1, int(depth)))
    stop = threading.Event()

    def put(item):
        while not stop.is_set():
            try:
                q.put(item, timeout=0.1)
                return True
            except queue.Full:
                pass
        return False

    def pump():
        it = iter(gen)
        end = _End()
        try:
            for b in it:
                if drop:
                    try:
                        q.put_nowait(b)
                    except queue.Full:
                        meta = getattr(b, "meta", None)
                        if meta is not None:
                            meta.stream.dropped += 1
                elif not put(b):
                    break
                if stop.is_set():
                    break
        except BaseException as e: # Re-raised in the consumer
            end = _End(e)
        finally:
            try:
                _close(it) # A generator source (a radio) cleans up in the thread that runs it
            except Exception:
                pass
        put(end)

    t = threading.Thread(target=pump, daemon=True, name=name)
    t.start()
    try:
        while True:
            item = q.get()
            if isinstance(item, _End):
                if item.exc is not None:
                    raise item.exc
                return
            yield item
    finally:
        stop.set()
        while t.is_alive(): # Unblock a producer waiting on a full queue
            try:
                q.get_nowait()
            except queue.Empty:
                t.join(timeout=0.05)

class Blocks:
    """Blocks of a stream. The operators apply to every block as it is produced, with state carried between them

        blocks + stage      # DSP stage or pipeline (also blocks * pipe, blocks / pipe)
        blocks >> sink      # write every block; returns the sink
        blocks.threaded()   # everything so far runs in a thread of its own
        blocks.pipelined(a + b + c) # a, b and c each in a thread of their own
    """
    def __init__(self, gen):
        self._gen = gen

    def __iter__(self):
        return iter(self._gen)

    def _map(self, fn):
        def gen(src=self._gen):
            try:
                for b in src:
                    yield fn(b)
            finally:
                _close(src) # A generator expression would leave the source (and its threads) to the garbage collector
        return Blocks(gen())

    def __add__(self, pipe):
        return self._map(lambda b: b + pipe)
    def __mul__(self, pipe):
        return self._map(lambda b: b * pipe)
    def __truediv__(self, pipe):
        return self._map(lambda b: b / pipe)

    def __rshift__(self, sink):
        try:
            for b in self._gen:
                sink.write(b)
        finally:
            _close(self._gen)
        return sink

    def threaded(self, depth=8, drop=False):
        """Run everything before this point (the source, the stages applied so far) in its own thread

        depth: blocks the queue to the next thread holds.
        drop: when the queue is full, discard the block and count it in `stream.dropped` instead of waiting. Use
            for a radio, which overruns anyway when it isn't read; the stream then has gaps (the stages carry on).
        An exception in the thread is raised where the blocks are read; leaving the loop stops the thread (and
        closes the source).
        """
        return Blocks(_threaded(self._gen, depth, drop, "rpps-blocks"))

    def pipelined(self, pipe, depth=4):
        """blocks + pipe, with every top-level stage of the pipe in its own thread (and the source in another)"""
        items = pipe.items if isinstance(pipe, base.Pipeline) else (pipe,)
        b = self.threaded(depth)
        for it in items:
            b = (b + it).threaded(depth)
        return b

    def collect(self):
        """All the output joined into one array"""
        try:
            parts = [np.asarray(b.data) for b in self._gen]
        finally:
            _close(self._gen)
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.complex64)

class Source:
    """Parent of the things that produce blocks. Call it with a block length for the blocks"""
    def __init__(self, Fs=1.0, cf=0.0, sps=0.0):
        self.Fs, self.cf, self.sps = Fs, cf, sps
        self.stream = Stream(Fs) # Shared by every block this source makes

    def _block(self, samples, flush=False):
        s = self.stream
        s.blocks += 1
        if self.Fs:
            s.air += len(samples) / self.Fs
        return Data(samples, flush=flush, Fs=self.Fs, sps=self.sps, cf=self.cf, stream=s)

    def _samples(self, count):
        raise NotImplementedError()

    def __call__(self, count):
        return Blocks(self._samples(count))

    def restart(self):
        """Start the stream over: the stages that ran on it forget their state"""
        self.stream.restart()

    def peek(self, count):
        """The first `count` samples as an array (not a block), without using them up: a File keeps its place"""
        raise StreamError(f"{type(self).__name__} can't peek")

    def __add__(self, pipe):
        raise StreamError("Give the source a block size first: source(4096) + stage")

class File(Source):
    """Samples from a file, in blocks (a short last block is delivered, flushing the stream)

    fmt: pysdrlib format name (see `rp.source.formats.ls()`)
    Fs, cf, sps: what the capture was recorded with
    scale: integer formats become floats in [-1, 1), as a device's samples are
    """
    def __init__(self, fmt, path, Fs=1.0, cf=0.0, sps=0.0, scale=True):
        super().__init__(Fs, cf, sps)
        self.fmt = fmts.get(fmt)
        self._f = pysdrlib.File(path, self.fmt.name, scale=scale)
        self.path = self._f.get_path()

    def __str__(self):
        return str(self._f)

    max_samp = property(lambda self: self._f.max_samp)
    cur_samp = property(lambda self: self._f.cur_samp)

    @property
    def percent(self):
        """Fraction of the file read [0, 1]"""
        return self._f.percent()

    def open(self):
        if self._f.f is None:
            self._f.open()

    def close(self):
        if self._f.f is not None:
            self._f.close()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *a):
        self.close()

    def seek(self, samp):
        self._f.seek(samp)

    def reset(self):
        self._f.reset()
        self.restart()

    def peek(self, count, start=0):
        """`count` samples from sample `start` (fewer near the end) as an array, leaving the read position where it was"""
        pos, was_open = self._f.cur_samp, self._f.f is not None
        self._f.seek(start)
        try:
            self.open()
            return self._writable(self._f.next(int(count), partial=True))
        finally:
            self._f.seek(pos)
            if not was_open:
                self.close()

    def read(self, count):
        """The next `count` samples as an array (not a block)"""
        self.open()
        return self._writable(self._f.next(int(count)))

    @staticmethod
    def _writable(x):
        return x if x.flags.writeable else x.copy()

    def _samples(self, count):
        with self._f:
            for x in self._f.forward(int(count), partial=True): # A short last block at the end
                yield self._block(self._writable(x), flush=self._f.cur_samp >= self._f.max_samp)

class Device(Source):
    """Samples from a pysdrlib device (RTL-SDR, HackRF, ..., "sim", or "remote"): blocks until the loop is left

        for blk in rp.source.Device("hackrf", cf=100e6, Fs=10e6)(8192) + rx: ...
    """
    def __init__(self, device, cf=None, Fs=None, rx_gain=None, sps=0.0, **open_args):
        """device: a pysdrlib device name, or a device you made (rp.source.Device(pysdrlib.device("sim", realtime=False))).
        rx_gain: the receive gain as a fraction 0-1 of the device's range (the same on any radio), or a {stage: fraction} dict"""
        self.sdr = pysdrlib.device(device) if isinstance(device, str) else device
        self.rx_gain = rx_gain
        self._open_args = open_args
        super().__init__(Fs if Fs is not None else self.sdr.get_rx_sample_rate(), cf if cf is not None else self.sdr.get_rx_freq(), sps)

    def peek(self, count):
        """The first `count` samples from the device, as an array. A device can't rewind: these are not in the blocks
        that follow (rp.analysis.survey keeps them in Survey.head)"""
        for blk in self._samples(count):
            return np.asarray(blk.data)

    def _samples(self, count):
        sdr = self.sdr
        sdr.open(**self._open_args)
        try:
            sdr.initialize(cf=self.cf, Fs=self.Fs, rx_gain=self.rx_gain)
            self.cf, self.Fs = sdr.get_rx_freq(), sdr.get_rx_sample_rate() # What the hardware settled on
            self.stream.src_Fs = self.Fs
            with sdr.stream(count, meta=True) as rx: # Exactly `count` samples a block; starts and stops receiving
                for x, m in rx:
                    self.stream.dropped += m.dropped
                    yield self._block(x)
        finally:
            if sdr.state["open"]:
                sdr.close()

class Sink:
    """Write blocks to a file with pysdrlib's FileSink: `block >> sink` or `blocks >> sink`

    The file is opened when the sink is made. Use it as a context manager (or `close()`) to finish the file.
    scale: samples in [-1, 1) are scaled up to the format's integers, which FileSink itself doesn't do
    clip: saturate what is out of range instead of failing (FileSink counts it in `clipped` and warns when closed)
    """
    def __init__(self, fmt, path, scale=True, append=False, clip=True):
        self.fmt = fmts.get(fmt)
        self.scale = self.fmt.full_scale if scale else 1.0
        self._sink = pysdrlib.FileSink(path, self.fmt.name, append=append, open=True, clip=clip)

    @property
    def path(self):
        return self._sink.path

    @property
    def count(self):
        return self._sink.count

    @property
    def clipped(self):
        """Samples that were saturated"""
        return self._sink.clipped

    def write(self, data):
        x = np.asarray(data.data if isinstance(data, base.data._Data) else data)
        if self.scale != 1.0:
            x = x * self.scale
        self._sink.write(x)
        return self

    def close(self):
        if self._sink.f is not None:
            self._sink.close()

    def __enter__(self):
        return self
    def __exit__(self, *a):
        self.close()
