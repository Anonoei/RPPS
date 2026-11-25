"""A Linux TUN/TAP interface, both ways: decoded packets onto it, packets written to it out to transmit

    tun = rp.source.Tun("rpps0")                                       # raw IP (a TUN): IPv4 and IPv6 packets
    tun = rp.source.Tun("rpps1", linktype="ieee802_15_4_nofcs")        # a TUN of another link type (TUNSETLINK)
    tun = rp.source.Tun("rpps2", mode="tap")                           # Ethernet frames (a TAP)
    p.capture.packets(rx.take_messages()) >> tun                         # receive: each packet of its link type
    for pkt in tun.packets(timeout=1.0): ...                             # transmit: what the system sends on it

Wireshark (or tcpdump) on the interface sees the packets as the system does, of the interface's link type. A TUN is raw
IP; on Linux it can take another link type (Linux's ARPHRD: 802.11, 802.11 with radiotap, 802.15.4 without FCS, AX.25
with its KISS byte), set while the interface is down, and then each packet carries packet information (its protocol,
the link type's `proto`) since the kernel takes a TUN's packets without it for IP only. A TAP is Ethernet. A packet of
a link type that is the interface's with transforms (config/linktype "from": 802.15.4 with its FCS on an 802.15.4
interface, which has none; AX.25 on Linux's AX.25 with its KISS byte) is converted; other packets are not written
(counted in `skipped`): a pcapng stream (rpps.source.PcapSink) carries any link type, and what each packet was.

Making the interface needs CAP_NET_ADMIN once; a persistent interface owned by the user is opened without it:

    sudo ip tuntap add dev rpps0 mode tun user $USER        # (mode tap for a TAP)
    sudo ip link set rpps0 up                               # (after the link type is set: Tun sets it while it is down)

`up`: bring the interface up when opened (True: or fail; None: try, and leave it to `ip link set up` without the right
to). Linux only: on Windows and macOS use a pcapng pipe (`PcapSink("pipe:<name>")`, a FIFO, or stdout).
"""
import errno
import os
import select
import socket
import struct
import sys

from ..errors import ConfigError

TUNSETIFF, TUNSETLINK = 0x400454CA, 0x400454CD
IFF_TUN, IFF_TAP, IFF_NO_PI = 0x0001, 0x0002, 0x1000
SIOCGIFFLAGS, SIOCSIFFLAGS, IFF_UP = 0x8913, 0x8914, 0x1
RAW_IP = ("raw", "ipv4", "ipv6") # What a plain TUN carries

class Tun:
    """A TUN or TAP interface (see module). name: the interface; mode: "tun" or "tap"; linktype: a TUN's link type
    (config/linktype: "raw" by default; one with an `arphrd`), a TAP's is "ethernet". count: packets written; skipped:
    packets of another link type, by its name"""
    def __init__(self, name, mode="tun", linktype=None, up=None):
        if not sys.platform.startswith("linux"):
            raise OSError("Tun: TUN/TAP interfaces are Linux's here; write a pcapng stream instead "
                          "(rp.source.PcapSink(\"pipe:<name>\") on Windows, a FIFO or \"-\" elsewhere)")
        from ..link.capture import linktype as _linktype
        if mode not in ("tun", "tap"):
            raise ConfigError(f"Tun {name}: mode is \"tun\" or \"tap\", not {mode!r}")
        lt = _linktype(linktype or ("raw" if mode == "tun" else "ethernet"))
        if mode == "tap" and lt.name != "ethernet":
            raise ConfigError(f"Tun {name}: a TAP carries Ethernet frames, not {lt.name} (make a TUN of that link type)")
        if mode == "tun" and lt.name not in RAW_IP and (lt.arphrd is None or lt.proto is None):
            raise ConfigError(f"Tun {name}: Linux has no interface type for {lt.name}; a TUN takes raw IP or one of the link "
                              "types with an `arphrd` and a `proto` (config/linktype); write a pcapng stream for the rest")
        self.name, self.mode, self.linktype = name, mode, lt
        self.pi = mode == "tun" and lt.name not in RAW_IP # (packet information: the protocol of a non-IP packet)
        flags = (IFF_TUN if mode == "tun" else IFF_TAP) | (0 if self.pi else IFF_NO_PI)
        self.fd = os.open("/dev/net/tun", os.O_RDWR)
        try:
            import fcntl
            fcntl.ioctl(self.fd, TUNSETIFF, struct.pack("16sH", name.encode(), flags))
            if self.pi:
                try:
                    fcntl.ioctl(self.fd, TUNSETLINK, lt.arphrd)
                except OSError as e:
                    if e.errno == errno.EBUSY:
                        raise OSError(f"Tun {name}: its link type is set while it is down: ip link set {name} down") from e
                    raise
        except BaseException:
            os.close(self.fd)
            raise
        self.count, self.skipped = 0, {}
        if up is not False:
            try:
                self.set_up(True)
            except PermissionError:
                if up:
                    raise
                from ..logger import Logger
                Logger().Child("Tun").warn(f"{name}: not brought up (no CAP_NET_ADMIN): sudo ip link set {name} up")

    def set_up(self, on=True):
        """Bring the interface up (or down); needs CAP_NET_ADMIN"""
        import fcntl
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            req = struct.pack("16sH", self.name.encode(), 0)
            got = struct.unpack("16sH", fcntl.ioctl(s, SIOCGIFFLAGS, req + bytes(22))[:18])[1]
            new = (got | IFF_UP) if on else (got & ~IFF_UP)
            fcntl.ioctl(s, SIOCSIFFLAGS, struct.pack("16sH", self.name.encode(), new) + bytes(22))

    def _carries(self, lt):
        return lt.name == self.linktype.name or (self.linktype.name == "raw" and lt.name in RAW_IP)

    def write(self, packets):
        """Write a Packet (rpps.link.capture), bytes (of the interface's link type), or an iterable of them; returns self"""
        from ..link.capture import Packet, convert
        items = [packets] if isinstance(packets, (Packet, bytes, bytearray)) else packets
        for p in items:
            if p is None:
                continue
            if isinstance(p, Packet):
                data = p.data
                if not self._carries(p.linktype):
                    try: # A link type that is this one with transforms (802.15.4 with its FCS): converted
                        data = convert(data, p.linktype, self.linktype)
                    except ValueError:
                        self.skipped[p.linktype.name] = self.skipped.get(p.linktype.name, 0) + 1
                        continue
            else:
                data = bytes(p)
            if self.pi:
                data = struct.pack("!HH", 0, self.linktype.proto) + data
            os.write(self.fd, data)
            self.count += 1
        return self

    def __rrshift__(self, packets):
        """packets >> tun"""
        return self.write(packets)

    def packets(self, timeout=None):
        """The packets the system sends on the interface, as bytes of its link type, until `timeout` seconds pass with none
        (None: for ever) or it is closed"""
        while self.fd is not None:
            r, _, _ = select.select([self.fd], [], [], timeout)
            if not r:
                return
            try:
                data = os.read(self.fd, 1 << 16)
            except OSError as e:
                if e.errno in (errno.EBADF, errno.EIO):
                    return
                raise
            yield data[4:] if self.pi else data

    def close(self):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None

    def __enter__(self):
        return self
    def __exit__(self, *a):
        self.close()
