"""Packets to samples through a profile's transmitter: what a capture rule gives on receive, sent back out

    enc = rp.profile.Encoder("ieee802154/2450", rate=250e3)
    for pkt in rp.source.Tun("rpps1", linktype="ieee802_15_4_nofcs").packets():   # what the system sends
        enc.packet(pkt) >> radio                                                   # (rp.source.DeviceSink)
    enc.flush() >> radio

A packet is the bytes a profile's capture rule (rpps.link.capture) makes of a message kind: the rule is undone (its
transforms in reverse), its included parts split back: a header part read into the fields of the frame (the header's
own schema), an FCS part dropped (the transmitter makes it), the payload sent with those fields over the profile's
defaults and the Encoder's. Each packet is one frame (a burst profile: one burst, placed by its burst step).
`bits(...)` sends bits as they are (`rpps tx` does), `gap` seconds of silence after each.
"""
import numpy as np

from ..errors import ConfigError

class Encoder:
    """A profile's transmitter, a packet (or a block of bits) at a time (see module)

    profile: a Profile or a name (params: its parameters); rate: the data's rate at the transmitter's first step (bits or
    symbols a second); kind: the message kind whose capture rule a packet follows (default: the one rule the profile has
    that captures, else its only kind); fields: the fields of every frame, over the profile's (a packet's own header over
    these). gap: seconds of silence after each packet or block (a profile without a burst step); hop, rate_out: a burst
    profile's carrier offsets cycled burst by burst and its output rate. linktype: the packets' link type when it isn't the
    rule's but one it converts from (802.15.4 without FCS off a Linux interface: the FCS is made). count: packets sent;
    Fs: the output's rate"""
    def __init__(self, profile, rate, kind=None, fields=None, gap=None, hop=None, rate_out=None, linktype=None, **params):
        from .profile import Profile, load
        from .burst import BurstTxRunner
        self.profile = profile if isinstance(profile, Profile) else load(profile, **params)
        self.rate, self.fields, self.gap = float(rate), dict(fields or {}), gap
        self.tx = self.profile.tx()
        self.bursts = isinstance(self.tx, BurstTxRunner)
        if self.bursts:
            bt = self.tx.burst_tx
            if gap is not None:
                bt.gap_s = gap
            if hop:
                bt.hop = [float(f) for f in hop]
            if rate_out:
                bt.Fs = rate_out
            bt.reset()
        elif hop or rate_out:
            raise ConfigError(f"{self.profile.name} has no burst step: hop and rate_out place bursts")
        self.kind, self.linktype = kind, linktype
        self._rule = self._op = None
        self.stream, self.Fs, self.count = None, None, 0

    # ---- bits

    def bits(self, bits, fields=None, last=False):
        """The samples of these bits (bool, MSB first; or bytes) sent as one frame, or burst"""
        import rpps as rp
        if isinstance(bits, (bytes, bytearray)):
            bits = np.unpackbits(np.frombuffer(bytes(bits), np.uint8)).astype(bool)
        f = {**self.fields, **(fields or {})} or None
        if self.bursts:
            o = self.tx(rp.Data(np.asarray(bits, bool), Fs=self.rate, sps=1), fields=f, flush=last)
            x = np.asarray(o.data, dtype=np.complex64)
        else:
            d = rp.Data(np.asarray(bits, bool), Fs=self.rate, sps=1, stream=self.stream, flush=True)
            self.stream = d.stream
            o = self.tx(d, fields=f)
            x = np.asarray(o.data, dtype=np.complex64)
            if self.gap:
                x = np.concatenate((x, np.zeros(int(round(self.gap*o.Fs)), np.complex64)))
        self.Fs = o.Fs
        self.count += 1
        return rp.Data(x, Fs=o.Fs)

    def flush(self):
        """A burst profile: the samples of the bursts not given yet (the stream ends); otherwise nothing"""
        import rpps as rp
        if not self.bursts:
            return rp.Data(np.zeros(0, np.complex64), Fs=self.Fs or 1.0)
        o = self.tx.flush()
        return rp.Data(np.asarray(o.data, dtype=np.complex64), Fs=o.Fs)

    # ---- packets

    def rule(self):
        """(the capture rule a packet follows, the step that emits its kind)"""
        if self._rule is None:
            p = self.profile
            kinds = p.kinds()
            kind = self.kind
            if kind is None:
                ruled = [k for k, r in p.capture.rules.items() if r is not False]
                kind = ruled[0] if len(ruled) == 1 else (next(iter(kinds)) if len(kinds) == 1 else None)
                if kind is None:
                    raise ConfigError(f"Encoder {p.name}: which kind's packets? kind= one of {sorted(kinds)}")
            if kind not in kinds:
                raise ConfigError(f"Encoder {p.name}: no message kind {kind!r} (gives {sorted(kinds)})")
            r = p.capture.rule(kind)
            if r is False:
                raise ConfigError(f"Encoder {p.name}: kind {kind!r} is not captured (\"capture\": false)")
            sid = kinds[kind]
            st = next(st for st in p.flat(p.steps) if st.id == sid)
            self._rule, self._op, self.kind = r, (st.obj if st.obj is not None else (st.rx[-1][1] if st.rx else None)), kind
        return self._rule, self._op

    def unpacket(self, data, linktype=None):
        """(payload bits, fields) of a packet (of `linktype`: the Encoder's, else the rule's): the capture rule undone,
        its parts split"""
        rule, op = self.rule()
        lt = linktype if linktype is not None else self.linktype
        if lt is not None and rule.linktype is not None:
            from ..link.capture import convert
            data = convert(data, lt, rule.linktype) # (packets of a link type the rule's is made from)
        body, fields = rule.unpacket(data)
        bits = np.unpackbits(np.frombuffer(body, np.uint8)).astype(bool)
        parts = list(rule.include)
        if "fcs" in parts:
            if parts[-1] != "fcs":
                raise ConfigError(f"Encoder {self.profile.name}: an included FCS is sent last")
            n = _fcs_bits(op)
            bits = bits[:len(bits) - n]
            parts.pop()
        if parts and parts[0] == "header":
            from ..link.header import Header
            if not isinstance(op, Header):
                raise ConfigError(f"Encoder {self.profile.name}: kind {self.kind!r} has no header step to read a header with")
            sent = op._wire(bits, op.order == "lsb") # (its own inverse: as sent)
            n = op.need(sent[:op.header_bits or len(sent)])
            values, ok = op.parse(sent[:n])
            if not ok:
                raise ValueError(f"Encoder {self.profile.name}: the packet's header fails its CRC")
            computed = {f["name"] for f in op.fields if "tx" in f}
            fields.update({k: v for k, v in values.items() if k in op.values and k not in computed})
            bits = bits[n:]
            parts.pop(0)
        if parts != ["payload"]:
            raise ConfigError(f"Encoder {self.profile.name}: a packet of parts {rule.include} can't be split back to its payload")
        return bits, fields

    def packet(self, data, fields=None, last=False, linktype=None):
        """The samples of one packet (bytes of the capture rule's link type, or of `linktype`), as one frame or burst"""
        bits, f = self.unpacket(data, linktype)
        return self.bits(bits, {**f, **(fields or {})}, last)

def _fcs_bits(op):
    if op is None:
        return 0
    if hasattr(op, "fcs_bits"):
        return int(op.fcs_bits)
    coder = getattr(op, "encoder", None)
    fcs = getattr(coder, "fcs", None)
    return int(fcs.width) if fcs is not None else 0
