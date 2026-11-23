import numpy as np

from ..code.types._code import _code

class bytestuff(_code):
    """Byte-stuffed framing (SLIP, KISS, asynchronous PPP): a flag byte between frames, and an escape byte before any
    flag or escape byte inside one, defined by config (see config/frame)

    flag: the frame delimiter (SLIP / KISS 0xC0, PPP 0x7E). escape: the escape byte (0xDB, 0x7D). escapes: {byte: what
    follows the escape for it} (SLIP: {0xC0: 0xDC, 0xDB: 0xDD}); or xor: the escaped byte is the byte XOR this (PPP 0x20),
    for the bytes in `escaped` (flag and escape when not given). fcs: a crc (or checksum) appended to each frame, or None.
    refin: bytes sent least significant bit first (a UART); the bit stream is in bytes from its first bit.
    min_bytes: shortest frame (with the FCS) kept.

    The same interface as `hdlc`: encode(frames) the bits of a list of byte frames; frames_at(bits) every frame between
    flags as (index of its opening flag's first bit, payload, fcs ok) and the aborts (an escape before a byte it doesn't
    escape); decode_stream(bits, held) for a stream cut anywhere.
    """
    def __init__(self, log, flag=0xC0, escape=0xDB, escapes=None, xor=None, escaped=None, fcs=None, refin=False, min_bytes=1):
        super().__init__(log, None, 1, 1)
        self.flag, self.escape, self.fcs, self.refin, self.min_bytes = flag, escape, fcs, refin, min_bytes
        if xor is not None:
            esc = escaped if escaped is not None else (flag, escape)
            self.escapes = {int(b): int(b) ^ xor for b in esc}
        else:
            self.escapes = {int(k): int(v) for k, v in (escapes or {flag: flag ^ 0x20, escape: escape ^ 0x20}).items()}
        self.unescape = {v: k for k, v in self.escapes.items()}
        if self.flag not in self.escapes or self.escape not in self.escapes:
            raise ValueError("byte stuffing: the flag and the escape byte must both be escaped")

    def _bits(self, data):
        return np.unpackbits(np.frombuffer(bytes(data), np.uint8), bitorder="little" if self.refin else "big")

    def _bytes(self, bits):
        return np.packbits(np.asarray(bits, dtype=np.uint8), bitorder="little" if self.refin else "big").tobytes()

    def frame(self, payload):
        """The bytes of one frame: flag, the stuffed payload and FCS, flag"""
        body = bytes(payload)
        if self.fcs is not None:
            body += self.fcs.to_bytes(self.fcs.compute(body))
        out = bytearray([self.flag])
        for b in body:
            if b in self.escapes:
                out += bytes([self.escape, self.escapes[b]])
            else:
                out.append(b)
        out.append(self.flag)
        return bytes(out)

    def encode(self, frames, idle=0):
        return self._bits(bytes([self.flag])*idle + b"".join(self.frame(f) for f in frames)).astype(bool)

    def _flags(self, raw):
        return np.nonzero(np.frombuffer(raw, np.uint8) == self.flag)[0]

    def frames_at(self, bits):
        """Frames as (index of the opening flag in `bits`, payload bytes, fcs ok), and the number of aborted frames"""
        b = np.asarray(bits, dtype=np.uint8)
        raw = self._bytes(b[:len(b)//8*8])
        flags = self._flags(raw)
        frames, aborts = [], 0
        for s, e in zip(flags[:-1], flags[1:]):
            if e - s <= 1:
                continue # Back-to-back flags: idle
            body, out, bad, esc = raw[s + 1:e], bytearray(), False, False
            for x in body:
                if esc:
                    if x not in self.unescape:
                        bad = True
                        break
                    out.append(self.unescape[x])
                    esc = False
                elif x == self.escape:
                    esc = True
                else:
                    out.append(x)
            if bad or esc:
                aborts += 1
                continue
            if len(out) < self.min_bytes:
                continue
            if self.fcs is None:
                frames.append((int(s)*8, bytes(out), True))
                continue
            n = self.fcs.width//8
            if len(out) <= n:
                aborts += 1
                continue
            ok = self.fcs.to_bytes(self.fcs.compute(bytes(out[:-n]))) == bytes(out[-n:])
            frames.append((int(s)*8, bytes(out[:-n]), ok))
        return frames, aborts

    def _decode(self, bits):
        frames, _ = self.frames_at(bits)
        good = [p for _, p, ok in frames if ok]
        out = np.unpackbits(np.frombuffer(b"".join(good), np.uint8)).astype(bool) if good else np.zeros(0, bool)
        return out, np.array([0 if ok else -1 for _, _, ok in frames], dtype=int)

    MAX_FRAME_BYTES = 1 << 13 # Bytes after a flag before the wait for a closing flag is given up

    def consumed(self, bits):
        """How many of these bits are done with: whole bytes up to the last flag (a frame may be open there), or all whole
        bytes when there is no flag (nothing can start one) or the frame after it has run too long"""
        n = len(bits)//8
        flags = self._flags(self._bytes(np.asarray(bits, dtype=np.uint8)[:n*8]))
        if not len(flags) or n - int(flags[-1]) > self.MAX_FRAME_BYTES:
            return n*8
        return int(flags[-1])*8

    def decode_stream(self, bits, held, frames=None):
        """Decode `held` (what the last call kept) and `bits` together: (data bits of the good frames, what to hold).
        frames: a list to append every frame found to, as `frames_at` gives them (index in held + bits)"""
        buf = np.concatenate((np.asarray(held, dtype=np.uint8), np.asarray(bits, dtype=np.uint8)))
        used = self.consumed(buf)
        part = buf[:used + 8] # the flag at `used` closes the frames before it
        out = self.decode(part) if used else np.zeros(0, bool)
        if frames is not None and used:
            frames += self.frames_at(part)[0]
        return out, buf[used:]
