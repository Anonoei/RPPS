"""Packets to Wireshark: pcapng written to a file, a FIFO, stdout or a Windows named pipe, live (`PcapSink`); and read back
(`Pcap`)

    rx, caps = p.rx(), p.capture                                    # a profile's capture rules (rpps.link.capture)
    with rp.source.PcapSink("-") as sink:                            # rpps rx ... --pcap - | wireshark -k -i -
        for block in src(65536):
            rx(block)
            caps.packets(rx.take_messages()) >> sink                 # (a list of Packets: each written)

Targets: a path (a file, or a FIFO made with `mkfifo`: opening waits for Wireshark), "-" for stdout, a file object, or
"pipe:<name>" on Windows: a named pipe server, `\\\\.\\pipe\\<name>`, that waits for `wireshark -k -i \\\\.\\pipe\\<name>`.
Every packet is flushed as it is written, so a live view keeps up.

Each (profile, kind, link type) is an interface of its own (named `rpps:<profile>:<kind>`, described with its link type
and frequency), so Wireshark's interface column and filters tell them apart. Each packet's comment says what it was
(`frame.comment contains "FAILED"`): profile and kind, the check and its verdict, frequency, quality, sample, fields; a
failed check also sets the packet's CRC-error flag (`frame.packet_flags_crc_error == 1`).

rpps: every packet wrapped in RPPS's own header on LINKTYPE_USER0, which tools/wireshark/rpps.lua shows as fields to
filter on (`rpps.crc_ok == 0`, `rpps.profile == "ble/adv"`, `rpps.field.type`) before handing the packet to its link
type's dissector. clock: "message" stamps a packet with its message's time (the source's, else seconds from the start
of the stream), "wall" with the time it is written.
"""
import json
import struct
import sys
import time as _time
from pathlib import Path

# pcapng blocks
SHB, IDB, EPB = 0x0A0D0D0A, 0x00000001, 0x00000006
BOM = 0x1A2B3C4D
OPT_END, OPT_COMMENT = 0, 1
IF_NAME, IF_DESCRIPTION, IF_TSRESOL = 2, 3, 9
SHB_USERAPPL = 4
EPB_FLAGS = 2
FLAG_INBOUND, FLAG_CRC_ERROR = 0x1, 1 << 24
USER0 = 147

def _opt(code, value):
    value = value.encode() if isinstance(value, str) else bytes(value)
    return struct.pack("<HH", code, len(value)) + value + bytes(-len(value) % 4)

def _block(kind, body):
    n = 12 + len(body)
    return struct.pack("<II", kind, n) + body + struct.pack("<I", n)

def shb(app="rpps"):
    return _block(SHB, struct.pack("<IHHq", BOM, 1, 0, -1) + _opt(SHB_USERAPPL, app) + _opt(OPT_END, b""))

def idb(dlt, name=None, description=None):
    opts = (_opt(IF_NAME, name) if name else b"") + (_opt(IF_DESCRIPTION, description) if description else b"")
    return _block(IDB, struct.pack("<HHI", dlt, 0, 0) + opts + _opt(IF_TSRESOL, b"\x09") + _opt(OPT_END, b""))

def epb(interface, t, data, comment=None, flags=FLAG_INBOUND):
    ns = int(round(t*1e9))
    body = struct.pack("<IIIII", interface, (ns >> 32) & 0xFFFFFFFF, ns & 0xFFFFFFFF, len(data), len(data))
    body += bytes(data) + bytes(-len(data) % 4)
    body += (_opt(OPT_COMMENT, comment) if comment else b"") + _opt(EPB_FLAGS, struct.pack("<I", flags)) + _opt(OPT_END, b"")
    return _block(EPB, body)

# RPPS's header on LINKTYPE_USER0 (tools/wireshark/rpps.lua): "RPPS", a version, the header's length, TLVs, the packet
RPPS_MAGIC, RPPS_VERSION = b"RPPS", 1
T_PROFILE, T_KIND, T_CHECK, T_CRC_OK, T_SAMPLE, T_TIME, T_FREQ, T_QUALITY, T_DLT, T_DISSECTOR, T_FIELDS = range(1, 12)

def rpps_header(p):
    """RPPS's header for a Packet: TLVs (type 1 byte, length 2 bytes big endian), then the packet's bytes"""
    from ..link.capture import _quality, _text
    def tlv(t, v):
        return struct.pack(">BH", t, len(v)) + v
    out = tlv(T_PROFILE, p.profile.encode()) + tlv(T_KIND, str(p.kind).encode())
    if p.check:
        out += tlv(T_CHECK, p.check.encode())
    out += tlv(T_CRC_OK, bytes([{True: 1, False: 0, None: 2}[p.crc_ok]]))
    out += tlv(T_SAMPLE, struct.pack(">q", int(p.sample)))
    if p.time is not None:
        out += tlv(T_TIME, struct.pack(">d", float(p.time)))
    if p.freq:
        out += tlv(T_FREQ, struct.pack(">d", float(p.freq)))
    q = _quality(p.quality)
    if q is not None:
        out += tlv(T_QUALITY, struct.pack(">d", q))
    out += tlv(T_DLT, struct.pack(">H", p.linktype.dlt))
    if p.linktype.dissector:
        out += tlv(T_DISSECTOR, p.linktype.dissector.encode())
    if p.fields:
        out += tlv(T_FIELDS, json.dumps({k: _text(v) for k, v in p.fields.items()}).encode())
    return RPPS_MAGIC + struct.pack(">BH", RPPS_VERSION, 7 + len(out)) + out + p.data

class PcapSink:
    """Write packets (rpps.link.capture.Packet) as pcapng: `packets >> sink`, `packet >> sink`, `sink.write(...)`

    target: a path or FIFO, "-" (stdout), a file object, or "pipe:<name>" (a Windows named pipe; see module).
    rpps: wrap every packet in RPPS's header (LINKTYPE_USER0, for tools/wireshark/rpps.lua). clock: "message" or "wall".
    count: packets written; per: {interface name: packets}"""
    def __init__(self, target, rpps=False, clock="message"):
        if clock not in ("message", "wall"):
            raise ValueError(f"PcapSink: clock is \"message\" or \"wall\", not {clock!r}")
        self.rpps, self.clock = bool(rpps), clock
        self._own = True
        if hasattr(target, "write"):
            self.f, self._own = target, False
        elif target == "-":
            self.f, self._own = sys.stdout.buffer, False
        elif isinstance(target, str) and target.startswith("pipe:"):
            self.f = _windows_pipe(target[5:])
        else:
            self.f = open(Path(target), "wb", buffering=0)
        self.target = target
        self._ifs = {} # (profile, kind, dlt) -> interface id
        self.count, self.per = 0, {}
        self._put(shb())

    def _put(self, b):
        self.f.write(b)
        if hasattr(self.f, "flush"):
            self.f.flush()

    def _interface(self, p, dlt):
        key = (p.profile, p.kind, dlt)
        i = self._ifs.get(key)
        if i is None:
            i = self._ifs[key] = len(self._ifs)
            name = f"rpps:{p.profile}:{p.kind}"
            desc = f"RPPS {p.profile} {p.kind}: {p.linktype.name}" + (f" at {p.freq/1e6:.6f} MHz" if p.freq else "")
            self._put(idb(dlt, name, desc))
            self.per[name] = 0
        return i

    def write(self, packets):
        """Write a Packet or an iterable of them (None items skipped); returns the sink"""
        from ..link.capture import Packet
        for p in ([packets] if isinstance(packets, Packet) else packets):
            if p is None:
                continue
            data, dlt = (rpps_header(p), USER0) if self.rpps else (p.data, p.linktype.dlt)
            i = self._interface(p, dlt)
            t = _time.time() if self.clock == "wall" or p.time is None else p.time
            flags = FLAG_INBOUND | (FLAG_CRC_ERROR if p.crc_ok is False else 0)
            self._put(epb(i, t, data, p.comment(), flags))
            self.count += 1
            self.per[f"rpps:{p.profile}:{p.kind}"] += 1
        return self

    def __rrshift__(self, packets):
        """packets >> sink (a list or generator of Packets)"""
        return self.write(packets)

    def close(self):
        if self.f is not None and self._own:
            self.f.close()
        self.f = None

    def __enter__(self):
        return self
    def __exit__(self, *a):
        self.close()

def _windows_pipe(name):
    """A Windows named pipe server, \\\\.\\pipe\\<name>, connected (this waits for a reader: Wireshark)"""
    if sys.platform != "win32":
        raise OSError(f"pipe:{name}: named pipes are Windows'; on Linux and macOS make a FIFO (mkfifo) or write to \"-\"")
    import ctypes
    import msvcrt
    from ctypes import wintypes
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateNamedPipeW.restype = wintypes.HANDLE
    k32.CreateNamedPipeW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.DWORD, wintypes.DWORD,
                                     wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID]
    k32.ConnectNamedPipe.argtypes = [wintypes.HANDLE, wintypes.LPVOID]
    PIPE_ACCESS_OUTBOUND, PIPE_TYPE_BYTE, PIPE_WAIT, ERROR_PIPE_CONNECTED = 0x2, 0x0, 0x0, 535
    h = k32.CreateNamedPipeW(f"\\\\.\\pipe\\{name}", PIPE_ACCESS_OUTBOUND, PIPE_TYPE_BYTE | PIPE_WAIT, 1, 1 << 16, 1 << 16, 0,
                             None)
    if h in (None, wintypes.HANDLE(-1).value):
        raise ctypes.WinError(ctypes.get_last_error())
    if not k32.ConnectNamedPipe(h, None) and ctypes.get_last_error() != ERROR_PIPE_CONNECTED:
        raise ctypes.WinError(ctypes.get_last_error())
    return open(msvcrt.open_osfhandle(h, 0), "wb", buffering=0)

class Record:
    """A packet read from a capture: dlt (its link type's number), data, time (seconds), comment, flags, interface"""
    __slots__ = ("dlt", "data", "time", "comment", "flags", "interface")

    def __init__(self, dlt, data, time, comment=None, flags=0, interface=0):
        self.dlt, self.data, self.time, self.comment, self.flags, self.interface = dlt, data, time, comment, flags, interface

    def __repr__(self):
        return f"<Record dlt {self.dlt} {len(self.data)} bytes @{self.time:.9f}{' ' + repr(self.comment) if self.comment else ''}>"

class Pcap:
    """Packets of a pcapng or pcap capture, in order: `for rec in rp.source.Pcap(path)` (Records). source: a path, bytes,
    or a file object. A FIFO is read as it is written. interfaces: [(dlt, name)] of a pcapng's interfaces seen so far"""
    def __init__(self, source):
        self.source = source
        self.interfaces = []

    def _open(self):
        if isinstance(self.source, (bytes, bytearray)):
            import io
            return io.BytesIO(bytes(self.source)), True
        if hasattr(self.source, "read"):
            return self.source, False
        return open(Path(self.source), "rb"), True

    def __iter__(self):
        f, own = self._open()
        try:
            head = _read(f, 4)
            if head is None:
                return
            magic = struct.unpack("<I", head)[0]
            if magic == SHB:
                yield from self._pcapng(f, head)
            elif magic in (0xA1B2C3D4, 0xD4C3B2A1, 0xA1B23C4D, 0x4D3CB2A1):
                yield from self._pcap(f, head)
            else:
                raise ValueError(f"Pcap: not a pcap or pcapng capture (starts {head.hex()})")
        finally:
            if own:
                f.close()

    def _pcap(self, f, head):
        e = "<" if head in (b"\xd4\xc3\xb2\xa1", b"\x4d\x3c\xb2\xa1") else ">"
        ns = head in (b"\x4d\x3c\xb2\xa1", b"\xa1\xb2\x3c\x4d")
        rest = _read(f, 20) # version (2, 2), zone, sigfigs, snaplen, link type
        dlt = struct.unpack(e + "I", rest[16:20])[0] & 0xFFFF
        self.interfaces = [(dlt, None)]
        while True:
            h = _read(f, 16)
            if h is None:
                return
            sec, frac, n, _ = struct.unpack(e + "IIII", h)
            data = _read(f, n) or b""
            yield Record(dlt, data, sec + frac*(1e-9 if ns else 1e-6))

    def _pcapng(self, f, head):
        e, res = "<", []
        block = head
        while True:
            if block is None:
                return
            kind = struct.unpack(e + "I", block)[0]
            raw = _read(f, 4)
            if raw is None:
                return
            if kind == SHB:
                bom = _read(f, 4)
                e = "<" if struct.unpack("<I", bom)[0] == BOM else ">"
                n = struct.unpack(e + "I", raw)[0]
                body = bom + _read(f, n - 16)
                self.interfaces, res = [], []
            else:
                n = struct.unpack(e + "I", raw)[0]
                body = _read(f, n - 12)
            _read(f, 4)
            if kind == IDB:
                dlt = struct.unpack(e + "H", body[:2])[0]
                opts = _options(body[8:], e)
                tsres = opts.get(IF_TSRESOL, [b"\x06"])[0][0]
                res.append(10.0**-tsres if not tsres & 0x80 else 2.0**-(tsres & 0x7F))
                name = opts.get(IF_NAME, [b""])[0].decode(errors="replace") or None
                self.interfaces.append((dlt, name))
            elif kind == EPB:
                i, hi, lo, cap, _ = struct.unpack(e + "IIIII", body[:20])
                data = body[20:20 + cap]
                opts = _options(body[20 + cap + (-cap % 4):], e)
                comment = opts.get(OPT_COMMENT, [None])[0]
                flags = struct.unpack(e + "I", opts[EPB_FLAGS][0])[0] if EPB_FLAGS in opts else 0
                yield Record(self.interfaces[i][0], data, ((hi << 32) | lo)*res[i],
                             comment.decode(errors="replace") if comment is not None else None, flags, i)
            block = _read(f, 4)

def _options(b, e):
    out, i = {}, 0
    while i + 4 <= len(b):
        code, n = struct.unpack(e + "HH", b[i:i + 4])
        if code == OPT_END:
            break
        out.setdefault(code, []).append(bytes(b[i + 4:i + 4 + n]))
        i += 4 + n + (-n % 4)
    return out

def _read(f, n):
    """n bytes (waiting on a FIFO), or None at the end"""
    out = b""
    while len(out) < n:
        b = f.read(n - len(out))
        if not b:
            return None if not out else out
        out += b
    return out
