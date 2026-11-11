"""Known structure among the symbols of a frame (ARCHITECTURE.md, symbol): physical headers, pilot blocks, phase scrambling

These work frame by frame. A frame starts wherever a profile applies a change to them (rpps.profile.control): a frame
sync publishes each frame's start, a header publishes what follows it, and each step restarts its frame there. Used on
their own, a stream starts on a frame.

    Header         known symbols (a preamble), then a short block code carrying fields: decoded and published on
                   receive, built from the frame's fields on transmit (DVB-S2 PLHEADER: SOF + PLS)
    Pilots         a block of known symbols after every few slots: removed on receive, inserted on transmit
    PhaseScramble  every symbol turned by a multiple of a step, the multiple from a binary sequence, restarting each
                   frame (DVB-S2 PL scrambling)

A header is a definition: config/mod/header/<name>.json, `rp.symbol.load("dvbs2_pl")`

    {"base": "header", "comment": "...", "ref": "...",
     "constellation": "bpsk", "phase": 45, "rotate": 90,      symbol k is the point turned by phase + k*rotate degrees,
     "cycle": 2,                                              k counted modulo cycle (pi/2-BPSK: 0, 90, 0, 90...)
     "known": {"bits": "0x18D2E82", "n": 26},                 the preamble, first bit first
     "code": {"n": 64, "rows": ["0x...", ...],                generator rows, one per message bit, n bits each, and
              "scramble": "0x..."},                           a mask XOR'd onto every code word
     "fields": [{"name": "modcod", "bits": 5}, ...],          the message, first field first, most significant bit first
     "max_errors": 16}                                        receive: a header with more bit errors is refused
"""
from types import SimpleNamespace

import numpy as np

from ..base import Mod, dtype
from ..base.types import Category, Rep
from ..errors import ConfigError
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
            from ..profile.control import anchor
            anchor(data.meta, self, data.meta.pos, 0)

    def restart(self, st):
        st.i = 0

    def _out(self, data, y, ctx):
        return data.derive(np.asarray(y, dtype=np.complex64), Category.SYMBOL, dtype.SYMBOLS, Mod, ctx, rep=Rep.SYMBOLS)

# ---- Physical header

class Header(_Frame):
    """A physical header at the start of every frame: known symbols, then a code word whose message is the fields

        rx: frame -> payload. Each frame's header is decoded (soft, every message tried: the code is short) and its fields
            published from the payload's first symbol (rp.profile.publish). A header with more than `max_errors` bit errors
            is refused, and the frame dropped. Its estimates are in stream.est[name]: frames, refused, last errors.
        tx: payload -> frame, the header built from the fields as last controlled (`control(modcod=..., ...)`).
    Frames start where a profile says (a frame sync before it), or at the start of the stream.
    """
    def __init__(self, name, constellation, phase, rotate, known, rows, n, scramble, fields, max_errors=None, cycle=None):
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
        return np.array([rotating(w, self.constellation, self.phase, self.rotate, len(self.known_bits), self.cycle)
                         for w in self.codebook])

    def encode_fields(self, values):
        """The header's symbols for these field values"""
        bits = np.concatenate([_bits(int(values.get(f, 0)), b) for f, b in self.fields])
        idx = int(bits @ (1 << np.arange(self.k - 1, -1, -1)))
        return np.concatenate((self.preamble, rotating(self.codebook[idx], self.constellation, self.phase, self.rotate,
                                                       len(self.known_bits), self.cycle)))

    def decode_fields(self, sym):
        """(fields, bit errors) of a header's symbols (the known part included)"""
        y = np.asarray(sym[len(self.known_bits):self.length], dtype=np.complex128)
        y = y*np.conj(turns(len(y), self.phase, self.rotate, self.cycle, len(self.known_bits)))
        llr = self.constellation.llr(y) # positive: bit 0
        score = (1.0 - 2.0*self.codebook) @ llr
        best = int(np.argmax(score))
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
        from ..profile.control import publish, anchor
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
                st.mode = "skip"
                continue
            est["frames"] += 1
            at = st.pos + self.length
            publish(data.meta, self, fields, at=at, out=n_out)
            anchor(data.meta, self, at, n_out)
            st.mode = "payload"
        y = np.concatenate(out) if out else x[:0]
        return self._out(data, y, ctx)

    def encode(self, data):
        from ..profile.control import anchor
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
                      obj["fields"], obj.get("max_errors"), obj.get("cycle"))

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
    dropped (counted in stream.est[name]["extra"]): the next frame starts where a profile says."""
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
    sequence z (a `scram` sequence definition), i counted from the frame start (and from 0 again every `length` symbols
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
            z = np.asarray(self.sequence.sequence(n + max(self.offsets)), dtype=np.int64)
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

    def encode(self, data):
        return self._run(data, True)

    def decode(self, data):
        return self._run(data, False)
