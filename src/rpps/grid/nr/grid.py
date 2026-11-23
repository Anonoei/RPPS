"""The NR resource grid of each slot: NrCell's slots to a stream of frames, one a slot

    HEAD items       cell id, slot of the frame 0-9, FFT size, slot count, the first symbol of the SS/PBCH block NrCell
                     tracks if the slot holds it (else -1), its index, its half frame
    14 x nfft        the grid G[l, bin]: every FFT bin, lowest frequency first, the block's centre at nfft/2

NR has no reference signal over the whole band: each channel after it (a branch of the profile) estimates its channel from
its own DM-RS. `unpack(frame)` gives the header and the grid.
"""
from types import SimpleNamespace

import numpy as np

from ...base import Stage
from ...base.types import Category, Rep
from . import phy

HEAD = 8

def frame_len(nfft):
    return HEAD + 14*nfft

def unpack(frame):
    f = np.asarray(frame)
    nfft = int(round(f[2].real))
    head = {"pci": int(round(f[0].real)), "slot": int(round(f[1].real)), "nfft": nfft, "count": int(round(f[3].real)),
            "ssb_symbol": int(round(f[4].real)), "i_ssb": int(round(f[5].real)), "n_hf": int(round(f[6].real))}
    return head, f[HEAD:HEAD + 14*nfft].reshape(14, nfft)


class NrGrid(Stage):
    """Slots of samples to grid frames (see the module); cell: the name of NrCell's estimates"""
    CATEGORY = Category.SYNC
    CONTROLS = {}

    def __init__(self, cell="cell", period=0.02):
        self.cell, self.period = cell, float(period)

    def __str__(self):
        return "NrGrid()"

    def takes(self, encode=True):
        return {Rep.SAMPLES}

    def new_state(self, meta):
        return SimpleNamespace(buf=None, count=0)

    def step(self, x, st, meta):
        fs = meta.Fs
        nfft = int(round(fs/phy.SCS))
        S = phy.slot_len(nfft)
        F = frame_len(nfft)
        meta.op.Fs, meta.op.sps = fs*F/S, 1
        x = np.asarray(x)
        st.buf = x if st.buf is None or not len(st.buf) else np.concatenate((st.buf, x))
        n = len(st.buf)//S
        out = np.zeros((n, F), np.complex64)
        cell = meta.stream.est.get(self.cell, {})
        pci, i_ssb, n_hf = cell.get("pci"), cell.get("i_ssb", 0), cell.get("n_hf", 0)
        per = max(int(round(self.period/0.01)), 1) # Frames a period
        for i in range(n):
            slot, frame = st.count % 10, st.count//10
            ssb = phy.SSB_SYMBOLS[i_ssb % 2] if (frame % per == 0 and slot == 5*n_hf + i_ssb//2) else -1
            out[i, :7] = (-1 if pci is None else pci), slot, nfft, st.count, ssb, i_ssb, n_hf
            out[i, HEAD:] = phy.grid(st.buf[i*S:(i + 1)*S], nfft).reshape(-1)
            st.count += 1
        st.buf = st.buf[n*S:]
        return out.reshape(-1)
