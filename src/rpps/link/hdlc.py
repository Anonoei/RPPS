import numpy as np

from ..code.types._code import _code

class hdlc(_code):
    """HDLC framing: flags, bit stuffing and an FCS, defined by config (see config/frame)

    fcs: a `crc` instance. flag: the flag byte (0x7E). stuff: ones before a zero is inserted (5)
    nrzi: line coding (a 0 is a transition, a 1 is no change). Bytes go out least significant bit first.
    min_bytes: shortest frame (including the FCS) accepted.

    encode(frames) takes a list of bytes. decode(bits) returns the frames whose FCS checks out; `last` holds
    (frames, bad frames, aborts) for the call.
    """
    def __init__(self, log, fcs, flag=0x7E, stuff=5, nrzi=False, min_bytes=4):
        super().__init__(log, None, 1, 1)
        self.fcs, self.flag, self.stuff, self.nrzi, self.min_bytes = fcs, flag, stuff, nrzi, min_bytes
        self.flag_bits = self._lsb(bytes([flag]))
        self.last = None

    @staticmethod
    def _lsb(data):
        return np.unpackbits(np.frombuffer(bytes(data), np.uint8), bitorder="little")

    def _stuffed(self, bits):
        out, ones = [], 0
        for b in bits:
            out.append(b)
            ones = ones + 1 if b else 0
            if ones == self.stuff:
                out.append(0)
                ones = 0
        return np.array(out, dtype=np.uint8)

    def frame(self, payload):
        """Bits of one frame: flag, stuffed payload + FCS, flag"""
        body = bytes(payload)
        body += self.fcs.to_bytes(self.fcs.compute(body))
        return np.concatenate((self.flag_bits, self._stuffed(self._lsb(body)), self.flag_bits))

    def encode(self, frames, idle=2):
        bits = np.concatenate([np.tile(self.flag_bits, idle)] + [self.frame(f) for f in frames] + [np.tile(self.flag_bits, idle)])
        if self.nrzi:
            bits = np.cumsum(1 - bits) % 2 # 0: toggle
        return bits.astype(bool)

    def deframe(self, bits):
        """Frames as (payload bytes, fcs ok) in the bit stream, and the number of aborted frames"""
        b = np.asarray(bits, dtype=np.uint8)
        if self.nrzi:
            b = np.concatenate(([0], (b[1:] == b[:-1]).astype(np.uint8)))
        # Flags: 01111110 patterns (LSB first: the flag byte's bits in transmission order)
        f = self.flag_bits
        n = len(f)
        win = np.lib.stride_tricks.sliding_window_view(b, n)
        starts = np.nonzero((win == f).all(axis=1))[0]
        frames, aborts = [], 0
        for s, e in zip(starts[:-1], starts[1:]):
            if e - s <= n:
                continue # Adjacent flags: idle
            body = b[s+n:e]
            out, ones, bad = [], 0, False
            for bit in body:
                if ones == self.stuff:
                    ones = 0
                    if bit == 0:
                        continue # Stuffed zero
                    bad = True # More than `stuff` ones in a row inside a frame: abort
                    break
                out.append(bit)
                ones = ones + 1 if bit else 0
            if bad or len(out) % 8:
                aborts += 1
                continue
            raw = np.packbits(np.array(out, dtype=np.uint8), bitorder="little").tobytes()
            if len(raw) < self.min_bytes:
                continue
            n_fcs = self.fcs.width // 8
            ok = self.fcs.to_bytes(self.fcs.compute(raw[:-n_fcs])) == raw[-n_fcs:]
            frames.append((raw[:-n_fcs], ok))
        return frames, aborts

    def decode(self, bits):
        frames, aborts = self.deframe(bits)
        good = [p for p, ok in frames if ok]
        self.last = (good, len(frames) - len(good), aborts)
        return np.concatenate([np.unpackbits(np.frombuffer(p, np.uint8)) for p in good]).astype(bool) if good else np.zeros(0, bool)
