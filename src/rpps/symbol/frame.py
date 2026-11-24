"""Known structure among the symbols of a frame (rpps.symbol): physical headers, pilot blocks, phase scrambling

These work frame by frame. A frame starts wherever a profile applies a change to them (rpps.profile.control): a frame
sync publishes each frame's start, a header publishes what follows it, and each step restarts its frame there. Used on
their own, a stream starts on a frame.

    Header         known symbols (a preamble), then a short block code carrying fields: decoded and published on
                   receive, built from the frame's fields on transmit (DVB-S2 PLHEADER: SOF + PLS)
    Pilots         a block of known symbols after every few slots: removed on receive, inserted on transmit
    PhaseScramble  every symbol turned by a multiple of a step, the multiple from a binary sequence, restarting each
                   frame (DVB-S2 PL scrambling)

A header is a definition: config/header/physical/<name>.json, `rp.symbol.load_header("dvbs2_pl")`

    {"base": "header", "type": "physical", "comment": "...", "ref": "...",
     "constellation": "bpsk", "phase": 45, "rotate": 90,      symbol k is the point turned by phase + k*rotate degrees,
     "cycle": 2,                                              k counted modulo cycle (pi/2-BPSK: 0, 90, 0, 90...)
     "known": {"bits": "0x18D2E82", "n": 26},                 the preamble, first bit first
     "code": {"n": 64, "rows": ["0x...", ...],                generator rows, one per message bit, n bits each, and
              "scramble": "0x..."},                           a mask XOR'd onto every code word
     "fields": [{"name": "modcod", "bits": 5}, ...],          the message, first field first, most significant bit first
     "turn": {"field": "s2x", "degrees": 90},                 optional: a message with this one-bit field set has its
                                                              code word sent turned this much more (DVB-S2X's MODCODs)
     "max_errors": 16}                                        receive: a header with more bit errors is refused
"""
from types import SimpleNamespace

import numpy as np

from ..base import Mod, dtype
from ..base.types import Category, Rep
from ..errors import ConfigError
from ..base.rpps import FRAME
from ..base.control import publish, anchor, emit
from .modulation import Modulation

def _bits(value, n):
    """n bits of a value (an int or a "0x..." string, any width), most significant first"""
    v = int(value, 0) if isinstance(value, str) else int(value)
    return np.array([(v >> (n - 1 - i)) & 1 for i in range(n)], dtype=np.uint8)

def turns(n, phase=0.0, rotate=0.0, cycle=None, start=0):
    """exp(j(phase + k*rotate)) in degrees for symbols k = start .. start+n-1, k modulo cycle when given"""
    k = start + np.arange(n)
    if cycle:
        k = k % int(cycle)
    return np.exp(1j*np.deg2rad(phase + rotate*k))

def rotating(bits, constellation, phase=0.0, rotate=0.0, start=0, cycle=None):
    """Symbols of bits on a constellation turned by `phase` degrees, and by `rotate` more each symbol (modulo `cycle`
    symbols), counting from `start` (pi/2-BPSK: "bpsk", 45, 90, cycle 2)"""
    sym = constellation.modulate(np.asarray(bits)).astype(np.complex128)
    return sym*turns(len(sym), phase, rotate, cycle, start)

class _Frame(Modulation):
    """A symbol step that restarts at every frame start (`control`)"""
    CONTROLS = {}
    SCOPE = FRAME
    _start = True # The stream starts on a frame

    def control(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        self._start = True

    def takes(self, encode):
        return {Rep.SYMBOLS}

    def pos_rate(self, decode):
        return 1.0

    def _state(self, ctx, enc):
        key = (self, "tx" if enc else "rx")
        st = ctx.get(key)
        st = SimpleNamespace(**vars(st)) if st is not None else SimpleNamespace(i=0, held=None, pos=None, mode="payload")
        ctx[key] = st
        self._began = self._start
        if self._start:
            self._start = False
            self.restart(st)
        return st

    _began = False

    def _anchor(self, data):
        """At a frame start: its first item is where the output goes on from (what came before may have been cut)"""
        if self._began:
            anchor(data.meta, self, data.meta.pos, 0)

    def restart(self, st):
        st.i = 0

    def _out(self, data, y, ctx):
        return data.derive(np.asarray(y, dtype=np.complex64), Category.SYMBOL, dtype.SYMBOLS, Mod, ctx, rep=Rep.SYMBOLS)

# ---- Physical header

class Header(_Frame):
    """A physical header at the start of every frame: known symbols, then a code word whose message is the fields

        rx: frame -> payload. Each frame's header is decoded (soft, every message tried: the code is short) and its fields
            published from the payload's first symbol (rp.base.control.publish). A header with more than `max_errors` bit errors
            is refused, and the frame dropped. Its estimates are in stream.est[name]: frames, refused, last errors. Each header
            is also emitted as a message (rpps.profile.message) from its first symbol, its fields and bit errors with it:
            crc_ok True when taken, False when refused.
        tx: payload -> frame, the header built from the fields as last controlled (`control(modcod=..., ...)`).
    Frames start where a profile says (a frame sync before it), or at the start of the stream.
    """
    EMITS = True # Every header, taken or refused, is a message (rpps.profile.message)

    def __init__(self, name, constellation, phase, rotate, known, rows, n, scramble, fields, max_errors=None, cycle=None,
                 turn=None):
        super().__init__()
        self.name = name
        self.constellation, self.phase, self.rotate, self.cycle = constellation, float(phase), float(rotate), cycle
        self.known_bits = np.asarray(known, dtype=np.uint8)
        self.fields = [(f["name"], int(f["bits"])) for f in fields]
        self.k = sum(b for _, b in self.fields)
        G = np.array([_bits(r, n) for r in rows], dtype=np.uint8)
        if G.shape[0] != self.k:
            raise ConfigError(f"header {name}: {G.shape[0]} code rows for {self.k} field bits")
        self.n = int(n)
        self.mask = np.zeros(self.n, np.uint8) if scramble is None else _bits(scramble, self.n)
        msgs = ((np.arange(2**self.k)[:, None] >> np.arange(self.k - 1, -1, -1)) & 1).astype(np.uint8)
        self.messages = msgs
        self.codebook = ((msgs @ G) % 2) ^ self.mask # (2^k, n)
        self.max_errors = self.n//4 if max_errors is None else int(max_errors)
        self.length = len(self.known_bits) + self.n # Symbols per header (one bit per symbol)
        self.preamble = rotating(self.known_bits, constellation, self.phase, self.rotate, 0, cycle)
        # turn: {"field": f, "degrees": d}: a message with that one-bit field set has its code word sent turned by d more
        # degrees (DVB-S2X: its MODCODs' PLS a quarter turn on from the SOF), so the field is read from the turn and the code
        self.turned = np.zeros(len(self.messages), bool)
        self.turn_deg = 0.0
        if turn:
            names = [f for f, _ in self.fields]
            if turn.get("field") not in names or dict(self.fields)[turn["field"]] != 1:
                raise ConfigError(f"header {name}: turn needs a one-bit field of {names}, not {turn.get('field')!r}")
            self.turned = self.messages[:, names.index(turn["field"])].astype(bool)
            self.turn_deg = float(turn.get("degrees", 0))
        self.CONTROLS = {"frame": "live", **{f: "live" for f, _ in self.fields}}
        self.values = {f: 0 for f, _ in self.fields}

    def __str__(self):
        return f"Header:{self.name}"

    def control(self, frame=None, **fields):
        unknown = set(fields) - set(self.values)
        if unknown:
            raise ConfigError(f"header {self.name}: no field {sorted(unknown)}")
        self.values = {**self.values, **{k: int(v) for k, v in fields.items() if v is not None}}
        self._start = True

    def words(self):
        """The symbols each message's code word is sent as, after the known part: (2^k, n)"""
        return np.array([rotating(w, self.constellation, self.phase + self.turn_deg*t, self.rotate, len(self.known_bits),
                                  self.cycle) for w, t in zip(self.codebook, self.turned)])

    def encode_fields(self, values):
        """The header's symbols for these field values"""
        bits = np.concatenate([_bits(int(values.get(f, 0)), b) for f, b in self.fields])
        idx = int(bits @ (1 << np.arange(self.k - 1, -1, -1)))
        return np.concatenate((self.preamble, rotating(self.codebook[idx], self.constellation,
                                                       self.phase + self.turn_deg*self.turned[idx], self.rotate,
                                                       len(self.known_bits), self.cycle)))

    def decode_fields(self, sym):
        """(fields, bit errors) of a header's symbols (the known part included)"""
        y = np.asarray(sym[len(self.known_bits):self.length], dtype=np.complex128)
        y = y*np.conj(turns(len(y), self.phase, self.rotate, self.cycle, len(self.known_bits)))
        llr = self.constellation.llr(y) # positive: bit 0
        score = (1.0 - 2.0*self.codebook) @ llr
        if self.turn_deg: # Turned messages scored on the symbols turned back
            llr_t = self.constellation.llr(y*np.exp(-1j*np.deg2rad(self.turn_deg)))
            score = np.where(self.turned, (1.0 - 2.0*self.codebook) @ llr_t, score)
        best = int(np.argmax(score))
        if self.turned[best]:
            llr = llr_t
        errors = int(np.sum((llr < 0) != self.codebook[best].astype(bool)))
        m = self.messages[best]
        out, i = {}, 0
        for f, b in self.fields:
            out[f] = int(m[i:i + b] @ (1 << np.arange(b - 1, -1, -1)))
            i += b
        return out, errors

    def restart(self, st):
        st.mode, st.held, st.pos = "header", np.zeros(0, np.complex128), None

    def decode(self, data):
        ctx = dict(data.ctx)
        st = self._state(ctx, False)
        x = np.asarray(data.data)
        pos0 = data.meta.pos
        est = data.meta.stream.est.setdefault(self.name, {"frames": 0, "refused": 0, "errors": None})
        out, n_out, i = [], 0, 0
        while i < len(x):
            if st.mode == "payload":
                out.append(x[i:])
                n_out += len(x) - i
                break
            if st.mode == "skip":
                break
            if st.pos is None:
                st.pos = pos0 + i
            take = min(self.length - len(st.held), len(x) - i)
            st.held = np.concatenate((st.held, x[i:i + take]))
            i += take
            if len(st.held) < self.length:
                break
            fields, errors = self.decode_fields(st.held)
            est["errors"] = errors
            if errors > self.max_errors:
                est["refused"] += 1
                emit(data.meta, self, None, {**fields, "errors": int(errors)}, None, False, st.pos)
                st.mode = "skip"
                continue
            est["frames"] += 1
            at = st.pos + self.length
            emit(data.meta, self, None, {**fields, "errors": int(errors)}, None, True, st.pos)
            publish(data.meta, self, fields, at=at, out=n_out)
            anchor(data.meta, self, at, n_out)
            st.mode = "payload"
        y = np.concatenate(out) if out else x[:0]
        return self._out(data, y, ctx)

    def encode(self, data):
        ctx = dict(data.ctx)
        x = np.asarray(data.data)
        start = self._start
        self._state(ctx, True)
        if start:
            anchor(data.meta, self, data.meta.pos, 0)
            x = np.concatenate((self.encode_fields(self.values), x))
        return self._out(data, x, ctx)

    @staticmethod
    def load(name, obj):
        from .serial import load as load_mod
        for k in ("constellation", "code", "fields"):
            if k not in obj:
                raise ConfigError(f"header {name}: needs '{k}'")
        known = obj.get("known") or {"bits": 0, "n": 0}
        code = obj["code"]
        return Header(name, load_mod(obj["constellation"]), obj.get("phase", 0), obj.get("rotate", 0),
                      _bits(known["bits"], int(known["n"])), code["rows"], code["n"], code.get("scramble"),
                      obj["fields"], obj.get("max_errors"), obj.get("cycle"), obj.get("turn"))

# ---- Pilot blocks

def pilot_layout(slots, pilots, slot=90, every=16, length=36):
    """(symbols in the frame, start of each pilot block): `slots` slots of `slot` symbols, a block of `length` pilots after
    every `every` slots but the last"""
    blocks = (slots - 1)//every if pilots else 0
    starts = [every*slot*(b + 1) + length*b for b in range(blocks)]
    return slots*slot + blocks*length, starts

class Pilots(_Frame):
    """Blocks of known pilot symbols among a frame's data: removed on receive, inserted on transmit

    The frame has `slots` slots of `slot` symbols; with `pilots` on, a block of `length` pilots (each `value`) follows
    every `every` slots but the last (DVB-S2: 90, 16, 36, (1+j)/sqrt2). On receive, symbols past the end of a frame are
    dropped (counted in `stream.est[name]["extra"]`): the next frame starts where a profile says."""
    CONTROLS = {"slots": "live", "pilots": "live"}

    def __init__(self, slot=90, every=16, length=36, value=(1 + 1j)/np.sqrt(2), slots=None, pilots=True, name="pilots"):
        super().__init__()
        self.slot, self.every, self.length, self.value = int(slot), int(every), int(length), complex(value)
        self.slots, self.pilots, self.name = slots, bool(pilots), name

    def __str__(self):
        return f"Pilots({self.slot}/{self.every}/{self.length})"

    def control(self, slots=None, pilots=None):
        if slots is not None:
            self.slots = int(slots)
        if pilots is not None:
            self.pilots = bool(pilots)
        self._start = True

    def layout(self):
        return pilot_layout(self.slots, self.pilots, self.slot, self.every, self.length)

    def pos_rate(self, decode):
        if self.slots is None:
            return 1.0
        total, _ = self.layout()
        r = self.slots*self.slot/total
        return r if decode else 1/r

    def _is_pilot(self, i):
        """Which frame positions i (array) are pilots, and which are past the frame"""
        total, starts = self.layout()
        pil = np.zeros(len(i), dtype=bool)
        for s in starts:
            pil |= (i >= s) & (i < s + self.length)
        return pil, i >= total

    def kernel(self, params, how):
        """The batch pilot remover (decoding), for the frame layout `params` set: pilots, and what is past the frame, dropped"""
        if how != "/":
            return None
        from .kernels import Gather
        if params:
            self.control(**params)
        if self.slots is None:
            return Gather(lambda L: np.ones(L, dtype=bool), str(self))
        total, starts = self.layout()
        def keep(L):
            pil = np.zeros(L, dtype=bool)
            for s in starts:
                pil[s:s + self.length] = True
            return ~pil[:L] & (np.arange(L) < total)
        return Gather(keep, str(self))

    def decode(self, data):
        ctx = dict(data.ctx)
        st = self._state(ctx, False)
        self._anchor(data)
        x = np.asarray(data.data)
        if self.slots is None:
            return self._out(data, x, ctx)
        i = st.i + np.arange(len(x))
        pil, past = self._is_pilot(i)
        st.i += len(x)
        if past.any():
            est = data.meta.stream.est.setdefault(self.name, {"extra": 0})
            est["extra"] = est.get("extra", 0) + int(past.sum())
        return self._out(data, x[~pil & ~past], ctx)

    def encode(self, data):
        ctx = dict(data.ctx)
        st = self._state(ctx, True)
        self._anchor(data)
        x = np.asarray(data.data)
        if self.slots is None or not self.pilots:
            return self._out(data, x, ctx)
        per = self.every*self.slot # Data symbols between blocks
        blocks = (self.slots - 1)//self.every
        out, j = [], 0
        while j < len(x):
            d = st.i # Data symbols of this frame so far
            nxt = (d//per + 1)*per
            take = min(len(x) - j, nxt - d)
            out.append(x[j:j + take])
            j += take
            st.i += take
            if st.i == nxt and st.i//per <= blocks and j <= len(x) and st.i < self.slots*self.slot:
                out.append(np.full(self.length, self.value, dtype=np.complex64))
        return self._out(data, np.concatenate(out) if out else x, ctx)

# ---- Phase scrambling

class PhaseScramble(_Frame):
    """Every symbol of a frame turned by step * R(i) degrees, R(i) = sum_j 2**(J-1-j) z(i + offsets[j]) for a binary
    sequence z (rpps.sequence), i counted from the frame start (and from 0 again every `length` symbols
    when that is set). DVB-S2 PL scrambling: z the Gold sequence, offsets (131072, 0), step 90 degrees.
    Receiving turns them back."""
    CONTROLS = {"length": "live"}

    def __init__(self, sequence, offsets=(0,), step=90.0, length=None, name="scramble"):
        super().__init__()
        self.sequence, self.offsets, self.step = sequence, [int(o) for o in offsets], float(step)
        self.length, self.name = length, name
        self._R = np.zeros(0, dtype=np.int64)

    def __str__(self):
        return f"PhaseScramble({self.name})"

    def control(self, length=None):
        self.length = None if length is None else int(length)
        self._start = True

    def R(self, n):
        """R(0..n-1)"""
        if len(self._R) < n:
            z = np.asarray(self.sequence.bits(n + max(self.offsets)), dtype=np.int64)
            J = len(self.offsets)
            self._R = sum((z[o:o + n] << (J - 1 - j)) for j, o in enumerate(self.offsets))
        return self._R[:n]

    def _run(self, data, enc):
        ctx = dict(data.ctx)
        st = self._state(ctx, enc)
        x = np.asarray(data.data)
        i = st.i + np.arange(len(x))
        if self.length:
            i = i % self.length
        st.i += len(x)
        r = self.R(int(i.max()) + 1)[i] if len(x) else np.zeros(0, np.int64)
        turn = np.exp((1j if enc else -1j)*np.deg2rad(self.step)*r)
        return self._out(data, x*turn, ctx)

    def kernel(self, params, how):
        """The batch rotation, for a frame that starts at its first symbol (`control` marks it) and restarts every `length`"""
        if how not in ("/", "*"):
            return None
        from .kernels import Rotate
        if params:
            self.control(**params)
        enc, step, length = how == "*", self.step, self.length
        def phasors(L):
            i = np.arange(L)
            if length:
                i = i % length
            r = self.R(int(i.max()) + 1)[i] if L else np.zeros(0, np.int64)
            return np.exp((1j if enc else -1j)*np.deg2rad(step)*r)
        return Rotate(phasors, self.name)

    def encode(self, data):
        return self._run(data, True)

    def decode(self, data):
        return self._run(data, False)

# ---- Known symbols before each frame

class Marker(_Frame):
    """Known symbols at the start of every frame (a symbol-level sync word: a Zadoff-Chu preamble, a word already mapped)

        rx: frame -> the symbols after the marker (a frame sync gives each frame from its preamble on); the marker's
            match, |<marker, received>| / (|marker| |received|), is in stream.est[name]["match"]
        tx: the marker, then the frame's symbols

    symbols: the marker; or `zc` (a length) and the Zadoff-Chu root and shift, controlled per frame (`root`, `shift`),
    or `sequence` (a definition with symbols), so a transmitter sends the preamble that says what its frame is.
    length: symbols of a frame after the marker; None: frames as a profile's control starts them (`frame`).
    """
    CONTROLS = {"frame": "live", "root": "live", "shift": "live", "sequence": "live"}

    def __init__(self, symbols=None, length=None, zc=None, root=None, shift=0, name="marker"):
        super().__init__()
        self.zc, self.root, self.shift, self.name = zc, root, int(shift or 0), name
        self.length = None if length is None else int(length)
        if self.length is not None and self.length <= 0:
            raise ValueError("symbol Marker: the frame length after the marker is positive (or None: frames by control)")
        self.symbols = None if symbols is None else np.asarray(symbols, dtype=np.complex128)
        if self.symbols is None:
            self._make()

    def __str__(self):
        return f"Marker({self.name})"

    def _make(self):
        if self.zc is not None and self.root is not None:
            from ..sequence import zadoff_chu
            self.symbols = zadoff_chu(self.root, self.zc, self.shift)
        if self.symbols is None:
            raise ValueError("symbol Marker: its symbols, a sequence, or a Zadoff-Chu length and root")

    def control(self, frame=None, root=None, shift=None, sequence=None):
        if sequence is not None:
            from .. import sequence as seq
            self.symbols = np.asarray(seq.load(sequence).symbols, dtype=np.complex128)
        if root is not None or shift is not None:
            self.root = self.root if root is None else int(root)
            self.shift = self.shift if shift is None else int(shift)
            self._make()
        self._start = True

    @property
    def n(self):
        return len(self.symbols)

    def pos_rate(self, decode):
        if self.length is None:
            return 1.0
        L, n = self.length, self.n
        return L/(L + n) if decode else (L + n)/L

    def encode(self, data):
        from ..base.rpps import align
        ctx = dict(data.ctx)
        x = np.asarray(data.data, dtype=np.complex64)
        if self.length is None:
            if len(x) and self._start:
                self._start = False
                anchor(data.meta, self, data.meta.pos, self.n) # The frame's first symbol follows the marker
                x = np.concatenate((self.symbols.astype(np.complex64), x))
            return self._out(data, x, ctx)
        x = align(ctx, (self, "tx"), x, self.length, pad=True)
        frames = x.reshape(-1, self.length)
        y = np.concatenate((np.broadcast_to(self.symbols.astype(np.complex64), (len(frames), self.n)), frames), axis=1)
        return self._out(data, y.reshape(-1), ctx)

    def _match(self, data, head):
        m = self.symbols[:len(head)]
        den = np.linalg.norm(m)*np.linalg.norm(head)
        data.meta.stream.est[self.name] = {"match": float(abs(np.vdot(m, head))/den) if den else 0.0}

    def decode(self, data):
        from ..base.rpps import align
        ctx = dict(data.ctx)
        x = np.asarray(data.data)
        if self.length is not None:
            x = align(ctx, (self, "rx"), x, self.n + self.length, pad=False)
            frames = x.reshape(-1, self.n + self.length)
            if len(frames):
                self._match(data, frames[-1, :self.n])
            return self._out(data, frames[:, self.n:].reshape(-1), ctx)
        key = (self, "strip") # The marker at each frame start is stripped (held while it is not all in yet)
        if self._start:
            self._start = False
            ctx[key] = np.zeros(0, dtype=x.dtype)
        held = ctx.pop(key, None)
        y = x
        if held is not None:
            head = np.concatenate((held, x))
            if len(head) < self.n and not ctx.get("flush"):
                ctx[key] = head
                y = x[:0]
            else:
                self._match(data, head[:self.n])
                y = head[self.n:]
                anchor(data.meta, self, data.meta.pos + self.n - len(held), 0) # The first symbol after the marker
        return self._out(data, y, ctx)

# ---- Periodic symbols of no interest (status symbols)

class Puncture(_Frame):
    """Symbols at fixed places of every period of a frame, removed on receive and inserted on transmit: P25's status
    symbol after every 35 (`every` 36, `at` [35]), counted from the frame start

        rx: the frame without them; tx: the frame with `value` put in (a status symbol's level, a known point), its
            last ones (after its last data symbol: P25's status symbol at 359 of 360) when the next frame starts or the
            stream ends
    """
    CONTROLS = {"frame": "live"}

    def __init__(self, every, at, value=0.0, name="puncture"):
        super().__init__()
        self.every = int(every)
        self.at = sorted({int(a) % self.every for a in at})
        if not self.at or len(self.at) >= self.every:
            raise ValueError(f"symbol Puncture: some, not all, of every {self.every} positions")
        self.value, self.name = complex(value), name
        self.keep = np.array([i for i in range(self.every) if i not in self.at]) # where the data of a period goes
        self._hole = np.zeros(self.every, bool)
        self._hole[self.at] = True

    def __str__(self):
        return f"Puncture({self.name}: {self.at} of {self.every})"

    def pos_rate(self, decode):
        r = len(self.keep)/self.every
        return r if decode else 1/r

    def restart(self, st):
        st.i = 0 # receive: input symbols of the frame so far; transmit: output symbols

    def decode(self, data):
        ctx = dict(data.ctx)
        st = self._state(ctx, False)
        self._anchor(data)
        x = np.asarray(data.data)
        i = st.i + np.arange(len(x))
        st.i += len(x)
        return self._out(data, x[~self._hole[i % self.every]], ctx)

    def _trailing(self, i):
        """Holes from frame position i to the end of its period: the last ones of a frame, after its last data symbol"""
        n = 0
        while (i + n) % self.every and self._hole[(i + n) % self.every]:
            n += 1
        return n

    def encode(self, data):
        ctx = dict(data.ctx)
        prev = ctx.get((self, "tx"))
        tail = self._trailing(prev.i) if self._start and prev is not None else 0 # The last frame's holes, now it is over
        st = self._state(ctx, True)
        if self._began:
            anchor(data.meta, self, data.meta.pos, tail) # This frame's first symbol follows them
        x = np.asarray(data.data)
        d = len(self.keep)
        n_done = (st.i//self.every)*d + int(np.count_nonzero(~self._hole[:st.i % self.every])) # data symbols placed so far
        k = n_done + np.arange(len(x))
        o = (k//d)*self.every + self.keep[k % d] # output position of each
        end = int(o[-1]) + 1 if len(x) else st.i
        if ctx.get("flush"): # The stream's last frame ends here
            end += self._trailing(end)
        y = np.full(tail + end - st.i, self.value, dtype=np.complex64)
        y[tail + o - st.i] = x
        st.i = end
        return self._out(data, y, ctx)
