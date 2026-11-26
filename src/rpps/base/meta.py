"""Rate information and stream state that travel with data

Users don't build these: give `rp.Data` an Fs / sps / cf and the stages take care of the rest.
"""
import threading
from collections import deque
from time import perf_counter

def forward(to, name, doc=None):
    """A read-only property that reads `name` of the object in attribute `to`: self.Fs is self.state.Fs"""
    return property(lambda self: getattr(getattr(self, to), name), doc=doc)

class Rate:
    """Sample rate (Fs), samples per symbol (sps) and centre frequency (cf)"""
    __slots__ = ("Fs", "sps", "cf")
    def __init__(self, Fs, sps, cf):
        self.set(Fs, sps, cf)

    def set(self, Fs, sps, cf):
        self.Fs, self.sps, self.cf = Fs, sps, cf

    def copy(self):
        return Rate(self.Fs, self.sps, self.cf)

    @property
    def Ts(self):
        return 1/self.Fs

    @property
    def sr(self):
        return self.Fs/self.sps

    @property
    def isps(self):
        return int(round(self.sps, 0))

    def __repr__(self):
        return f"Rate(Fs={self.Fs}, sps={self.sps}, cf={self.cf})"

class Stream:
    """Everything that follows one source across its blocks, shared by every block (and every data derived from it)

    states: the carried state of each stage (filter history, oscillator phase, loop filters...), keyed by the stage,
        so a stage object is reusable and the same stage in two streams has two states
    det: detection result (baud, modulation, timing, cfo...), see rpps.sync.detect
    busy: seconds each operator spent on this stream's blocks (a stage's `step`, a pipe's encode or decode), its own time
        only (what it ran inside it is the inner one's); air: seconds of signal the source produced. busy/air (`load`)
        is the operator's share of real time: 1.0 exactly keeps up, over 1 it can't. `realtime()` is the one number
    lanes: {thread: how many run its work at once} (a parallel region's workers); a thread not named is one
    thread_of: {operator: the thread it ran on}: what `realtime` adds up per thread
    dropped: source samples lost: a radio's overruns (`Device`, as pysdrlib reports them) and blocks a threaded source
        discarded because the stages fell behind (`Blocks.threaded(drop=True)`, counted at the source's rate)
    pos: items the source has produced so far: the absolute position of its next block
    t0: time [s] of the source's item 0, when the source knows it (a radio's timestamp); None otherwise
    out_pos: items each stage has produced so far in this stream: the absolute position of its next output block
    in_pos: the position each stage expects its next input block at; one further on is a gap, which the stage carries
        into its own output positions
    est: what sync stages estimated, by name ({"align": {...}}), for whoever wants to know (det stays Acquire's)
    control: the control path (rpps.base.control.Control): fields header steps published, position anchors, and
        what a profile runner did with them (applied, re-run, dropped). None until a step publishes
    rx: how the source's samples were received (a pysdrlib `RxContext`: the chain and its LO, whether it inverts, the
        reference and its correction), None when the source doesn't know; rx_conjugated: the source conjugated them
        because the chain inverts (rpps.source.Source). Both survive `restart`

    Threads (Blocks.threaded): each stage's state is used only by the thread running that stage. `det` and
    `src_sps` have one writer (Acquire); `gen` lets blocks already on their way pick up a new sps (Meta.refresh). `det`
    here is the latest: a stage reads its block's (`Meta.det`), which Acquire stamps, so a stage running behind Acquire
    in another thread uses the detection its block went through with, not one made for a later block.
    """
    __slots__ = ("states", "det", "blocks", "src_Fs", "src_sps", "gen", "busy", "air", "dropped", "pos", "t0", "out_pos", "in_pos", "est", "control",
                 "thread_of", "lanes", "_snaps", "rx", "rx_conjugated")
    def __init__(self, Fs=None):
        self.states = {}
        self.det = None
        self.blocks = 0 # Blocks the source has produced
        self.src_Fs = Fs # Rate of the source
        self.src_sps = None # sps at the source, once a stage has learned it (Acquire): the next blocks start from it
        self.gen = 0 # Bumped when src_sps changes
        self.busy = {}
        self.thread_of = {}
        self.lanes = {}
        self._snaps = deque(maxlen=1024) # (air, busy) every SNAP seconds of signal: what `load(window)` measures over
        self.air = 0.0
        self.dropped = 0
        self.pos = 0
        self.t0 = None
        self.out_pos = {}
        self.in_pos = {}
        self.est = {}
        self.control = None
        self.rx, self.rx_conjugated = getattr(self, "rx", None), getattr(self, "rx_conjugated", False) # (kept by restart)

    SNAP = 0.02 # [s] of signal between the snapshots `load(window)` looks back through (1024 of them: 20 s)

    def add_air(self, seconds):
        """The source made `seconds` more of signal (a source calls this per block)"""
        self.air += seconds
        if not self._snaps or self.air - self._snaps[-1][0] >= self.SNAP:
            self._snaps.append((self.air, dict(self.busy))) # (dict() of a dict is one step under the GIL: safe while stages run)

    def load(self, window=None):
        """Share of real time each operator takes, {operator: busy/air}: 1.0 exactly keeps up, over 1 it falls behind.

        window: None, since the stream started; seconds, over about the last `window` seconds of signal (what a live view
        wants: a stage that slows down shows at once)"""
        busy, air = dict(self.busy), self.air
        if window is not None and self._snaps:
            a0, b0 = self._snaps[0]
            for a, b in reversed(self._snaps):
                if air - a >= window:
                    a0, b0 = a, b
                    break
            if air - a0 > 0:
                return {op: (t - b0.get(op, 0.0))/(air - a0) for op, t in busy.items()}
        return {op: t/air for op, t in busy.items()} if air > 0 else {}

    def realtime(self, window=None):
        """The stream's load as one number: per thread, the sum of the loads of what ran on it (divided by the thread's
        `lanes`), and the busiest thread's. Serial work is one thread (the sum); `Blocks.pipelined` gives a thread per
        stage (the slowest); a parallel region counts its workers' time over their number. <= 1 keeps up in real time.
        0.0 before the source has made any signal"""
        per = {}
        for op, v in self.load(window).items():
            t = self.thread_of.get(op)
            per[t] = per.get(t, 0.0) + v
        return max((v/self.lanes.get(t, 1) for t, v in per.items()), default=0.0)

    def restart(self):
        """Forget everything but the source's rate: the next block starts a new stream"""
        gen = self.gen + 1
        self.__init__(self.src_Fs)
        self.gen = gen

class Meta:
    """The rate information of one block of data, as a stage sees it

        Fs, sps, cf, isps, Ts: the rates of the block coming in (read-only for stages)
        op: the rates of the block going out. A stage changes these (meta.op.Fs /= 2), the carrier applies them
        delay: group delay of the data so far [s]. A stage adds its own
        stream: the Stream this block belongs to (stage state, detection)
        pos: absolute position of the block's first item in the stream, counted in items at this block's rate. It does
            not depend on how the stream was cut into blocks, so estimators can place their windows by it. Samples a
            source lost (a radio's overrun, a dropped block) are counted: positions stay true to time across a gap
        timed: `pos` counts this block's own items (a source's block, a stage's output), so a jump in it is a gap; a
            pipe's output keeps its input's position and is not timed
        flush: this block ends the stream: a stage gives out what it was holding back
        det: the detection this block went through Acquire with (the stream's, `stream.det`, until an Acquire stamps one)
    """
    __slots__ = ("state", "op", "delay", "stream", "gen", "pos", "timed", "flush", "_det")
    def __init__(self, Fs=1.0, sps=0.0, cf=0.0, delay=0.0, stream=None, pos=0):
        self.stream = Stream(Fs) if stream is None else stream
        if self.stream.src_Fs is None:
            self.stream.src_Fs = Fs
        if self.stream.src_sps is not None:
            sps = self.stream.src_sps # A stage already learned the source's sps
        self.state = Rate(Fs, sps, cf)
        self.op = Rate(Fs, sps, cf)
        self.delay = delay
        self.gen = self.stream.gen
        self.pos = int(pos)
        self.timed = False
        self.flush = False
        self._det = None

    @property
    def det(self):
        return self.stream.det if self._det is None else self._det

    @det.setter
    def det(self, det):
        """Acquire: the detection for this block and those made from it; the stream keeps the latest"""
        self._det = det
        self.stream.det = det

    def time_of(self, k=0):
        """Source time [s] of item k of this block: the stream's t0, plus the position at this rate, less the group delay.
        None when the source gave no time anchor"""
        t0 = self.stream.t0
        return None if t0 is None else t0 + (self.pos + k)/self.state.Fs - self.delay

    Fs, sps, cf = forward("state", "Fs"), forward("state", "sps"), forward("state", "cf")
    isps, Ts, sr = forward("state", "isps"), forward("state", "Ts"), forward("state", "sr")

    def fork(self):
        """Meta for the data a stage produces: the output rates become the input rates. Stream is shared"""
        m = Meta.__new__(Meta)
        m.stream = self.stream
        m.state = self.op.copy()
        m.op = self.op.copy()
        m.delay = self.delay
        m.gen = self.gen
        m.pos = self.pos
        m.timed = False
        m.flush = False
        m._det = self._det
        return m

    def refresh(self):
        """Pick up an sps learned after this block left the source

        One thread: never needed, every block starts after the last one finished. Threaded, blocks already between
        the source and Acquire carry the old sps: scale the source's new one by the Fs ratio, as the stages would have.
        """
        s = self.stream
        if self._det is not None: # Past Acquire: its sps is the one Acquire gave this block, not a later block's
            self.gen = s.gen
            return
        if self.gen != s.gen:
            self.gen = s.gen
            if s.src_sps is not None and s.src_Fs:
                sps = s.src_sps * self.state.Fs / s.src_Fs
                self.state.sps = sps
                self.op.sps = sps

    def calibrate_sps(self, sps):
        """Declare the sps at the current stage of the chain

        Blocks come from the source with the source's sps, so the sps at the source is worked out through the
        Fs ratio (decimation/resampling scale both) and handed to the blocks that follow.
        """
        self.stream.src_sps = sps * self.stream.src_Fs / self.state.Fs
        self.stream.gen += 1
        self.gen = self.stream.gen
        self.state.sps = sps
        self.op.sps = sps

    def state_of(self, stage):
        """Get (creating on first use) the carried state of a stage"""
        try:
            return self.stream.states[stage]
        except KeyError:
            st = self.stream.states[stage] = stage.new_state(self)
            return st

    def __str__(self):
        return f"Meta: Fs{self.state.Fs}, sps{self.state.sps}, cf{self.state.cf}"

_tls = threading.local()

def timed(stream, op, fn, *args):
    """fn(*args), its time charged to `op` in `stream.busy`: its own time only (an operator timed inside it takes its own
    off), so nothing is counted twice. stream None: run, not charged"""
    if stream is None:
        return fn(*args)
    stack = _tls.__dict__.setdefault("stack", [])
    stack.append(0.0)
    t = perf_counter()
    try:
        return fn(*args)
    finally:
        spent = perf_counter() - t
        inner = stack.pop()
        if stack:
            stack[-1] += spent
        busy = stream.busy
        busy[op] = busy.get(op, 0.0) + spent - inner
        stream.thread_of[op] = threading.get_ident()
