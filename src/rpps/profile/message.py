"""Decoded messages: what a profile receives, one record a frame, for a program to show or keep (README.dev.md, Profiles)

A step that knows where a frame starts and whether it checked out (a link header, an LTE channel) emits it with
rp.base.control.emit; the runner places it in the stream it was given and keeps it until `runner.take_messages()`:

    rx = rp.profile.load("adsb/mode_s").rx()
    for block in src(65536):
        rx(block)
        for m in rx.take_messages():
            print(m.kind, m.crc_ok, m.fields, m.payload.hex() if m.payload else "")

rp.profile.Decoder does this, and brings the samples to the profile's rate first.
"""
import copy

import numpy as np

class Message:
    """One decoded frame

    profile: the profile's name; kind: the step that emitted it (its id: "frame", "pdu", "pbch", "pdsch"...)
    fields: what the step decoded of the frame (header fields, a MIB, a DCI...)
    payload: the payload as bytes, most significant bit first (the last byte padded with zeros), None without one;
        bits: its length in bits
    crc_ok: whether the frame's check passed (False: shown so, fields as far as they were decoded), None without a check
    sample: where the frame starts, in samples of the stream the runner was given (a Decoder: of the samples it was fed),
        counted from the start of that stream
    time: seconds: the source's time of that sample when it has one (a radio's timestamp), else sample/Fs
    freq: the signal's centre frequency [Hz] when known (the stream's cf, a Decoder's cf + offset), else None
    quality: the profile's measure of its symbols when it has one (`stream.est["quality"]`: Es/N0, EVM), else None
    """
    __slots__ = ("profile", "kind", "fields", "payload", "bits", "crc_ok", "sample", "time", "freq", "quality")

    def __init__(self, profile, kind, fields=None, payload=None, bits=0, crc_ok=None, sample=0, time=None, freq=None,
                 quality=None):
        self.profile, self.kind, self.fields = profile, kind, dict(fields or {})
        self.payload, self.bits, self.crc_ok = payload, int(bits), crc_ok
        self.sample, self.time, self.freq, self.quality = int(sample), time, freq, quality

    @staticmethod
    def payload_of(bits):
        """(bytes, number of bits) of a bit array (bool, 0/1, or soft values: negative is a one), MSB first"""
        if bits is None:
            return None, 0
        b = np.asarray(bits)
        if b.dtype.kind == "f":
            b = b < 0
        b = b.astype(bool).ravel()
        return np.packbits(b).tobytes(), len(b)

    def to_dict(self):
        """Plain values, for JSON"""
        q = self.quality
        if q is not None and not isinstance(q, dict):
            q = {k: getattr(q, k) for k in getattr(q, "__slots__", ()) if isinstance(getattr(q, k), (int, float))}
        return {"profile": self.profile, "kind": self.kind, "fields": {k: _plain(v) for k, v in self.fields.items()},
                "payload": None if self.payload is None else self.payload.hex(), "bits": self.bits,
                "crc_ok": self.crc_ok, "sample": self.sample, "time": self.time, "freq": self.freq, "quality": q}

    def copy(self):
        return copy.copy(self)

    def placed(self, sample, Fs, cf=None, t0=None):
        """A copy placed in an outer stream: at `sample` there, of rate Fs (time from its t0), cf for freq when unknown"""
        m = copy.copy(self)
        m.sample = int(round(sample))
        m.time = (t0 or 0.0) + m.sample/Fs
        if m.freq is None and cf:
            m.freq = cf
        return m

    def __repr__(self):
        ok = {True: "ok", False: "FAILED", None: "-"}[self.crc_ok]
        f = ", ".join(f"{k}={v}" for k, v in self.fields.items())
        return f"<Message {self.profile} {self.kind} @{self.sample} {ok}: {f}{f' + {self.bits} bits' if self.bits else ''}>"

def _plain(v):
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (bytes, bytearray)):
        return bytes(v).hex()
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, dict):
        return {k: _plain(x) for k, x in v.items()}
    return v
