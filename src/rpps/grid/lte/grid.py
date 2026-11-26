"""The LTE resource grid of each subframe, with every port's channel on every resource element

LteGrid takes what LteCell gives (whole half-frames from a subframe 0, carrier offset off) and gives one frame per
subframe: a stream of complex items, `frame_len(n_rb)` a subframe, laid out as

    HEAD items      the subframe's header: cell id, subframe number 0..9, N_RB, subframe count, 1 for TDD
                    (as real values)
    14 x 12 N_RB    the grid G[l, k]: symbol l, subcarrier k from the lowest (the DC subcarrier left out)
    4 x 14 x 12 N_RB  the channel H[p, l, k] of ports 0-3, from the cell-specific reference signals

so the physical channels after it (each a branch of the profile, reading the same frames) take their resource elements
and undo the transmit diversity by `unpack`. The cell comes from LteCell's estimates in the stream.
"""
from types import SimpleNamespace

import numpy as np

from ...base import Stage
from ...base.types import Category, Rep
from . import phy

HEAD = 8

def frame_len(n_rb):
    return HEAD + 5*14*12*n_rb

def n_rb_of(frame_items):
    """N_RB from the frame length"""
    n, r = divmod(frame_items - HEAD, 5*14*12)
    if r:
        raise ValueError(f"{frame_items} items is not an LTE grid frame")
    return n

def unpack(frame):
    """(header dict, G (14, 12 N_RB), H (4, 14, 12 N_RB)) of one frame: views, no copies"""
    f = np.asarray(frame)
    n_rb = int(round(f[2].real))
    nsc = 12*n_rb
    head = {"pci": int(round(f[0].real)), "sf": int(round(f[1].real)), "n_rb": n_rb, "count": int(round(f[3].real)),
            "duplex": "tdd" if f[4].real > 0.5 else "fdd"}
    G = f[HEAD:HEAD + 14*nsc].reshape(14, nsc)
    H = f[HEAD + 14*nsc:HEAD + 5*14*nsc].reshape(4, 14, nsc)
    return head, G, H

def frames(x, n_rb):
    """The whole frames of a block: (n, frame_len) view"""
    F = frame_len(n_rb)
    x = np.asarray(x)
    return x[:len(x)//F*F].reshape(-1, F)


class LteGrid(Stage):
    """Subframes of samples to grid frames (see the module). n_rb: resource blocks (default: those that fill the FFT);
    cell: the name of LteCell's estimates"""
    CATEGORY = Category.SYNC
    CONTROLS = {}

    def __init__(self, n_rb=None, cell="cell"):
        self.n_rb, self.cell = n_rb, cell

    def __str__(self):
        return "LteGrid()"

    def takes(self, encode=True):
        return {Rep.SAMPLES}

    def new_state(self, meta):
        return SimpleNamespace(buf=None, count=0)

    def step(self, x, st, meta):
        fs = meta.Fs
        nfft = int(round(fs/phy.SCS))
        n_rb = self.n_rb or phy.N_RB[nfft]
        S = phy.subframe_len(nfft)
        F = frame_len(n_rb)
        meta.op.Fs, meta.op.sps = fs*F/S, 1
        x = np.asarray(x)
        st.buf = x if st.buf is None or not len(st.buf) else np.concatenate((st.buf, x))
        n = len(st.buf)//S
        out = np.zeros((n, F), np.complex64)
        cell = meta.stream.est.get(self.cell, {})
        pci, tdd = cell.get("pci"), cell.get("duplex") == "tdd"
        for i in range(n):
            sf = st.count % 10
            G = phy.grid(st.buf[i*S:(i + 1)*S], nfft, n_rb)
            out[i, :5] = (-1 if pci is None else pci), sf, n_rb, st.count, tdd
            out[i, HEAD:HEAD + G.size] = G.reshape(-1)
            if pci is not None:
                out[i, HEAD + G.size:] = phy.channel(G, pci, n_rb, sf).reshape(-1)
            st.count += 1
        st.buf = st.buf[n*S:]
        return out.reshape(-1)
