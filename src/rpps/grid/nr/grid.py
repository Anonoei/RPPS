"""The NR resource grid of each subframe: NrCell's slots to a stream of frames, one a millisecond

    HEAD items       cell id, subframe of the frame 0-9, SIZE (samples of a subframe / 15: nfft x 2^mu), subframe count,
                     the first slot symbol (counted through the subframe) of the SS/PBCH block NrCell tracks if the
                     subframe holds it (else -1) and the symbol past its last, its index, its half frame, mu of the
                     channels' grid, mu of the block's
    4 x 240          the block's own grid B[l, k] (its spacing: NR's may differ from the channels'), where held
    14 2^mu x nfft   the grid G[slot, l, bin] of each of the subframe's slots at the channels' spacing (the MIB's):
                     every FFT bin, lowest frequency first, the block's centre at nfft/2

NR has no reference signal over the whole band: each channel after it (a branch of the profile) estimates its channel from
its own DM-RS. A frame's length depends on the sample rate alone, 16 + 960 + 14 x Fs / 15 kHz. `unpack(frame)` gives the
header and the grids.
"""
from types import SimpleNamespace

import numpy as np

from ...base import Stage
from ...base.types import Category, Rep
from . import phy

HEAD = 16
BLOCK = 4*240

def frame_len(size):
    return HEAD + BLOCK + 14*size

def unpack(frame):
    """(header dict, G (slots, 14, nfft), B (4, 240)) of one frame: views, no copies"""
    f = np.asarray(frame)
    v = lambda i: int(round(f[i].real))
    size, mu = v(2), v(7)
    nfft = size >> mu
    head = {"pci": v(0), "subframe": v(1), "nfft": nfft, "count": v(3), "ssb_symbol": v(4), "ssb_end": v(5),
            "i_ssb": v(6), "mu": mu, "mu_ssb": v(8), "n_hf": v(9)}
    return head, f[HEAD + BLOCK:HEAD + BLOCK + 14*size].reshape(1 << mu, 14, nfft), f[HEAD:HEAD + BLOCK].reshape(4, 240)


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
        size = int(round(fs/15e3))
        S = 15*size # samples of a subframe
        F = frame_len(size)
        meta.op.Fs, meta.op.sps = fs*F/S, 1
        x = np.asarray(x)
        st.buf = x if st.buf is None or not len(st.buf) else np.concatenate((st.buf, x))
        n = len(st.buf)//S
        out = np.zeros((n, F), np.complex64)
        if not n:
            return out.reshape(-1)
        cell = meta.stream.est.get(self.cell, {})
        pci, i_ssb, n_hf = cell.get("pci"), cell.get("i_ssb", 0), cell.get("n_hf", 0)
        mu_ssb = phy.mu_of(cell.get("scs", 15e3))
        mu = phy.mu_of(cell.get("scs_common", cell.get("scs", 15e3)))
        nfft, nfft_ssb = size >> mu, size >> mu_ssb
        block = self._block(cell.get("pattern", "A"), i_ssb, n_hf, mu_ssb, mu, nfft_ssb, nfft)
        per = max(int(round(self.period*100)), 1)*10 # Subframes a period
        for i in range(n):
            sf = st.count % 10
            holds = pci is not None and st.count % per == block.subframe
            out[i, :10] = ((-1 if pci is None else pci), sf, size, st.count, block.first if holds else -1,
                           block.end if holds else -1, i_ssb, mu, mu_ssb, n_hf)
            y = st.buf[i*S:(i + 1)*S]
            if holds:
                W = np.fft.fftshift(np.fft.fft(y[block.starts[:, None] + np.arange(nfft_ssb)[None, :]], axis=1), axes=1)
                out[i, HEAD:HEAD + BLOCK] = (W*block.turn)[:, phy.ssb_bins(nfft_ssb)].reshape(-1)
            n_c = phy.slot_len(nfft)
            for s in range(1 << mu):
                at = HEAD + BLOCK + s*14*nfft
                out[i, at:at + 14*nfft] = phy.grid(y[s*n_c:(s + 1)*n_c], nfft, mu).reshape(-1)
            st.count += 1
        st.buf = st.buf[n*S:]
        return out.reshape(-1)

    @staticmethod
    def _block(pattern, i_ssb, n_hf, mu_ssb, mu, nfft_ssb, nfft):
        """Where block i_SSB of half frame n_hf is: its subframe, the FFT windows of its symbols in it (and the phase ramp
        their early start leaves), and the symbols of the channels' grid it overlaps, counted through the subframe"""
        if phy.SSB_PATTERNS[pattern][0] != mu_ssb:
            pattern = "C" if mu_ssb else "A"
        slot, l0 = phy.ssb_place(pattern, i_ssb)
        slot += (5 << mu_ssb)*n_hf
        subframe, within = divmod(slot, 1 << mu_ssb)
        cp = phy.cp_lengths(nfft_ssb, mu_ssb)[1]
        early = cp//4
        at = within*phy.slot_len(nfft_ssb) + int(phy.symbol_starts(nfft_ssb, mu_ssb)[l0])
        starts = at + np.arange(4)*(nfft_ssb + cp) - early
        turn = np.exp(2j*np.pi*(np.arange(nfft_ssb) - nfft_ssb//2)*early/nfft_ssb)
        begin, end = at - cp, at + 3*(nfft_ssb + cp) + nfft_ssb # (the block's samples)
        st = np.concatenate([s*phy.slot_len(nfft) + phy.symbol_starts(nfft, mu) for s in range(1 << mu)])
        lead = phy.cp_lengths(nfft, mu)[1]
        hit = np.nonzero((st + nfft > begin + lead//2) & (st - lead < end - lead//2))[0]
        return SimpleNamespace(subframe=subframe, starts=starts, turn=turn, first=int(hit[0]), end=int(hit[-1]) + 1)
