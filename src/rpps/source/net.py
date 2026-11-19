"""Samples from the network: raw IQ over UDP or TCP, in any pysdrlib format

    src = rp.source.Socket("cf64", "udp://0.0.0.0:5004", Fs=2e6, cf=100e6)       # listen for datagrams
    src = rp.source.Socket("ci16t", "tcp://10.0.0.5:1234", Fs=2.4e6)             # connect and read a byte stream
    for out in src(16384) + rx:
        ...

The bytes are cut into whole samples whatever the datagrams or reads hold (a sample split across two keeps its first
part until the rest comes), so the blocks are the same however the network delivers them. TCP ends when the far side
closes (the last block flushes the stream); UDP when `timeout` seconds pass with nothing received (None: never).
"""
import io
import socket
from urllib.parse import urlparse

import numpy as np

from .source import Source
from . import formats as fmts

class Socket(Source):
    """Raw samples from `url`: udp://host:port (bound and listened on) or tcp://host:port (connected to)

    fmt: pysdrlib format of the samples; scale: integer formats become floats in [-1, 1), as File's do.
    timeout: seconds of silence that end a UDP stream (None: wait for ever); a TCP stream ends when it is closed"""
    def __init__(self, fmt, url, Fs=1.0, cf=0.0, sps=0.0, scale=True, timeout=None, chunk=1 << 16):
        super().__init__(Fs, cf, sps)
        u = urlparse(url)
        if u.scheme not in ("udp", "tcp") or u.port is None:
            raise ValueError(f"Socket: {url!r} is not udp://host:port or tcp://host:port")
        self.url, self.proto, self.addr = url, u.scheme, (u.hostname or "0.0.0.0", u.port)
        self.fmt = fmts.get(fmt)
        self.scale = (1.0/self.fmt.full_scale) if (scale and self.fmt.integer) else 1.0
        self.timeout, self.chunk = timeout, int(chunk)

    def __str__(self):
        return f"Socket({self.fmt.name}, {self.url})"

    def _open(self):
        if self.proto == "udp":
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(self.addr)
        else:
            s = socket.create_connection(self.addr)
        s.settimeout(self.timeout)
        return s

    def _recv(self, s):
        """Bytes from the socket; b"" at the end of the stream"""
        try:
            return s.recv(self.chunk) if self.proto == "tcp" else s.recvfrom(self.chunk)[0]
        except socket.timeout:
            return b""

    def decode(self, raw):
        """Whole samples of `raw` bytes (as an array) and the bytes left over"""
        size = self.fmt.bytes
        n = len(raw)//size
        x = self.fmt.read(io.BytesIO(raw[:n*size]), n) if n else np.zeros(0, np.complex64)
        x = np.asarray(x)
        return (x*self.scale if self.scale != 1.0 else x), raw[n*size:]

    def _samples(self, count):
        count = int(count)
        s = self._open()
        buf, pending, n = b"", [], 0
        try:
            while True:
                data = self._recv(s)
                end = not data
                x, buf = self.decode(buf + data)
                if len(x):
                    pending.append(x)
                    n += len(x)
                while n >= count or (end and n):
                    allx = np.concatenate(pending)
                    out, rest = allx[:count], allx[count:]
                    pending, n = ([rest] if len(rest) else []), len(rest)
                    yield self._block(out.copy(), flush=end and n == 0)
                if end:
                    return
        finally:
            s.close()
