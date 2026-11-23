"""Batch kernels of the link operators: `HeaderDecode` of a framed header (rpps.parallel)

Checked row for row against the operator run frame by frame (tests/unit/test_batch_kernels.py). The header bits of each
row come to the host (a few bytes) to be read; the payloads are cut on whichever `xp` the batch is on
"""
import numpy as np

from ..base.data import Digital
from ..base.kernel import Form, Kernel, to_host
from ..base.types import Category, Rep, dtype
from ..errors import StreamError

class HeaderDecode(Kernel):
    """A framed header (`rpps.link.header.Header`) of fixed size, one frame a row: its fields and CRC read, and the payload
    they announce given (`payload`, at most the frame's `length` less the header). A row whose header fails its CRC is
    dropped (the operator gives it no payload, and a message with crc_ok False); one shorter than its header gives nothing,
    as the operator does. Rows of one batch give payloads of one length (as `LoopKernel` asks)"""
    def __init__(self, op, length):
        self.op, self.length = op, int(length)
        self.name = f"header:{op.name}"

    def shape(self, n_in):
        return None # (the payload's length is in each header)

    def run(self, xp, b):
        op = self.op
        R, L = b.x.shape
        if L > self.length:
            raise StreamError(f"{self.name}: a row of {L} bits holds more than one frame of {self.length}")
        hb = op.header_bits
        bits = (b.x < 0) if b.form.dt == dtype.LLR else b.x.astype(bool)
        head = to_host(bits[:, :hb]) if L >= hb else None
        lengths = {}
        for i in range(R):
            if b.bad[i]:
                continue
            if head is None: # (shorter than its header, as a frame cut short: no header is read, nothing given)
                lengths[i] = 0
                continue
            fields, ok = op.parse(head[i])
            if not ok:
                b.fail(i, f"{self.name}: header CRC failed")
                continue
            n = op.payload_bits(fields)
            n = min(n if n is not None else self.length, self.length - hb - op.fcs_bits, L - hb)
            lengths[i] = max(int(n), 0)
        if len(set(lengths.values())) > 1:
            raise StreamError(f"{self.name}: frames of one batch gave different lengths {sorted(set(lengths.values()))}")
        n = next(iter(lengths.values()), 0)
        y = bits[:, hb:hb + n] if head is not None else bits[:, :0]
        b.x = xp.ascontiguousarray(y)
        b.form = Form(Digital, Category.LINK, dtype.BITS, Rep.PACKETS, *b.form[4:])
        return b
