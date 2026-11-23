"""5G NR cell search and slot timing through LEO Doppler (38.211 7.4.2-7.4.3, 38.213 4.1): samples to whole slots

Acquiring (the first `acquire` seconds): the PSS at 3.84 MS/s over carrier offsets `offset` +- `span` (a LEO pass at
S-band moves the carrier by up to some 50 kHz), folded on the SS/PBCH block period (`period`, 20 ms). Then each block at
the full rate: the residual offset from the PSS-to-SSS phase, less the cell's bias in it (NR gives each symbol's carrier a
phase of its own, 38.211 5.4: a constant for a cell, found against the four symbols' cyclic prefixes), the SSS's N_ID1, and from the PBCH DM-RS the
block's index and half frame, which place the slots and the frame. Over the blocks of the acquisition, the sample clock
(the delay's rate: a satellite's range stretches the signal by tens of ppm) and the offset's rate.

Tracking: every block, its PSS's timing error and its offset (the bias kept up to date) drive two loops of the second order: time
(the next slot's start and the clock's rate) and frequency (the offset and its rate: a pass changes it by hundreds of Hz
a second). Between blocks the offset follows its rate, the derotation's phase continuous. With no Doppler the rates
stay near zero: the same receiver serves a terrestrial cell.

Out come the samples of whole slots from the start of the frame holding the first block acquired, derotated, each slot
started at the nearest sample to where the clock puts it (a sample dropped or repeated between slots), so slot n of the
output is slot n mod 10 of a frame, and every second frame holds the block (20 ms). Lost for `lose` periods, the cell is
acquired again, zeros standing in for the slots missed. `est["cell"]`: the cell, block, half frame, offset and its rate,
the clock's rate [ppm].
"""
from types import SimpleNamespace

import numpy as np

from ...base import Stage
from ...base.types import Category, Rep
from . import phy

def pss_time(n_id2, nfft):
    X = np.zeros(nfft, complex)
    X[(np.arange(127) - 64) % nfft] = phy.pss(n_id2) # SS/PBCH subcarriers 56-182 about its centre (120)
    return np.fft.ifft(X)

def _correlate(x, t):
    from ...signal.filters import convolve
    return convolve(x, np.conj(np.asarray(t)[::-1]), "valid")

def _lowpass(fs, cutoff=0.96e6, taps=129):
    h = np.sinc(2*cutoff/fs*(np.arange(taps) - (taps - 1)/2))*np.hamming(taps)
    return h/h.sum()


class NrCell(Stage):
    """NR cell search and slot timing, LEO Doppler tracked (see the module)

    offset, span, resolution: carrier offsets searched [Hz]; period: SS/PBCH block period [s]; acquire: seconds looked
    at; lose: periods without the block before acquiring again; gate: the PSS's normalized correlation for a block;
    fc: the block centre's absolute frequency (on the raster), when known: the PSS-to-SSS bias is then one of two, none
    or what the upconversion phase gives at fc (est "phase"), whichever the cyclic prefixes' average is nearer"""
    CATEGORY = Category.SYNC
    CONTROLS = {}

    def __init__(self, offset=0.0, span=60e3, resolution=4e3, period=0.02, acquire=0.065, lose=8, gate=0.25, fc=0.0, name="cell"):
        self.offset, self.span, self.resolution, self.period = float(offset), float(span), float(resolution), float(period)
        self.acquire, self.lose, self.gate, self.name = float(acquire), int(lose), float(gate), name
        self.fc = float(fc)

    def __str__(self):
        return f"NrCell({self.offset/1e3:+g} kHz +- {self.span/1e3:g})"

    def takes(self, encode=True):
        return {Rep.SAMPLES}

    def new_state(self, meta):
        return SimpleNamespace(buf=None, base=0, lock=None, est={"locked": False, "acquired": 0, "lost": 0, "slots": 0},
                               out_slots=0)

    # ---- one block, at the full rate

    def _measure(self, y, p, nfft, fs, n_id2, n_id1=None):
        """At the PSS whose useful part starts at y[p]: (SSS's N_ID1, residual offset [Hz], SSS metric, Y of the block's 4
        symbols (4, 240), the channel at the PSS (127))"""
        cp = phy.cp_lengths(nfft)[1]
        step = nfft + cp
        early = cp//4
        b = phy.ssb_bins(nfft)
        Y = []
        for s in range(4):
            a = p + s*step - early
            if a < 0 or a + nfft > len(y):
                return None
            W = np.fft.fftshift(np.fft.fft(y[a:a + nfft]))*np.exp(2j*np.pi*(np.arange(nfft) - nfft//2)*early/nfft)
            Y.append(W[b])
        Y = np.array(Y)
        h = Y[0, 56:183]*phy.pss(n_id2) # (PSS values are +-1)
        z = Y[2, 56:183]*np.conj(h)
        if n_id1 is None:
            m = np.abs(phy.sss_table(n_id2) @ z)/np.sqrt(np.sum(np.abs(z)**2)*127 + 1e-30)
            n_id1 = int(np.argmax(m))
            metric = float(m[n_id1])
        else:
            metric = float(np.abs(phy.sss(n_id1, n_id2) @ z)/np.sqrt(np.sum(np.abs(z)**2)*127 + 1e-30))
        # Two views of the residual offset: the phase from the PSS to the SSS (127 subcarriers each: little noise, but moved
        # by the phase NR's upconversion gives each symbol, 38.211 5.4, a constant for a cell), and each symbol's cyclic
        # prefix against its end (that phase cannot move it; few samples: noisy). The bias is their difference, averaged
        acc = 0j
        for k in range(4):
            u = p + k*step
            acc += np.sum(y[u - cp:u]*np.conj(y[u - cp + nfft:u + nfft]))
        f_cp = float(-np.angle(acc)/(2*np.pi*nfft/fs))
        ph = float(np.angle(np.sum(z*phy.sss(n_id1, n_id2))))
        return n_id1, (ph, f_cp, 2*step/fs), metric, Y, h

    @staticmethod
    def _offset(meas, bias):
        """The residual offset [Hz] of a block from the PSS-to-SSS phase, the cell's bias (a unit vector) taken off"""
        ph, _, span = meas
        return float(np.angle(np.exp(1j*ph)*np.conj(bias))/(2*np.pi*span))

    @staticmethod
    def _bias(meas):
        ph, f_cp, span = meas
        return np.exp(1j*(ph - 2*np.pi*f_cp*span))

    def _snap(self, raw, span):
        """The cell's bias from its running average `raw`: with fc, the nearer of none and the upconversion phase's"""
        if not self.fc:
            return raw/abs(raw) if abs(raw) > 0 else 1.0 + 0j, None
        term = np.exp(-2j*np.pi*((self.fc*span) % 1.0))
        applied = abs(raw - term) < abs(raw - 1.0)
        return (term if applied else 1.0 + 0j), bool(applied)

    def _ibar(self, Y, h, pci):
        """i_SSB + 4 n_hf from the PBCH DM-RS: the hypothesis its equalized values agree with best"""
        pk, pl, dk, dl = phy.ssb_map(pci)
        kk = np.arange(56, 183)
        H = np.interp(dk, kk, h.real) + 1j*np.interp(dk, kk, h.imag)
        r = Y[dl, dk]*np.conj(H)
        # Per symbol, the magnitude: a residual offset turns each symbol's phase from the PSS's by its own amount
        score = [float(sum(np.abs(np.sum((r*np.conj(phy.pbch_dmrs(pci, i)))[dl == l])) for l in (1, 2, 3))) for i in range(8)]
        return int(np.argmax(score)), score

    def _find(self, y, guess, tmpl, half):
        a = guess - half
        seg = y[max(a, 0):a + 2*half + len(tmpl)]
        if a < 0 or len(seg) < 2*half + len(tmpl):
            return None
        c = np.abs(_correlate(seg, tmpl))
        j = int(np.argmax(c))
        e = np.sqrt(np.sum(np.abs(seg[j:j + len(tmpl)])**2)*np.sum(np.abs(tmpl)**2))
        return a + j, float(c[j]/max(e, 1e-30))

    # ---- acquisition

    def _acquire(self, x, fs, nfft):
        # The samples moved by `offset`, filtered to the PSS's +-0.96 MHz and the span and taken to 3.84 MS/s once; each
        # offset tried is then a turn of the decimated samples
        from ...signal.filters import convolve
        ratio = int(round(fs/3.84e6))
        P = int(round(self.period*3.84e6))
        y0 = convolve(x*phy.lte_phy.ramp(len(x), 0.0, self.offset/fs), _lowpass(fs, 0.96e6 + self.span + 60e3), "same")[::ratio]
        # The correlations in the frequency domain: the samples' spectrum once, each offset a roll of it, each template's
        # conjugate spectrum once: one inverse FFT an offset and root
        n = len(y0)
        nf = 1 << int(np.ceil(np.log2(n + 256)))
        Y = np.fft.fft(y0, nf)
        T = [np.conj(np.fft.fft(pss_time(n2, 256), nf)) for n2 in range(3)]
        # The templates hold nothing past +-0.96 MHz, so each product's middle half is all of it: inverse FFTs of half the
        # size give the correlation at 1.92 MS/s (positions to 2 samples at 3.84 MS/s; the full-rate search refines them)
        q = nf//4
        T = [np.concatenate((t[-q:], t[:q])) for t in T]
        P2 = P//2
        best = None
        for d in np.arange(-self.span, self.span + self.resolution/2, self.resolution):
            k = int(round(d*nf/3.84e6))
            Yd = np.concatenate((Y[(np.arange(-q, 0) + k) % nf], Y[(np.arange(0, q) + k) % nf]))
            for n2 in range(3):
                c = np.abs(np.fft.ifft(np.fft.ifftshift(Yd*T[n2]))[:(n - 255)//2])**2 # (c[k] = sum y[2k + i] conj(t[i]))
                m = len(c)//P2*P2
                if not m:
                    continue
                fold = c[:m].reshape(-1, P2).mean(0)
                score = fold.max()/max(np.median(fold), 1e-30)
                if best is None or score > best[3]:
                    best = (n2, self.offset + d, int(np.argmax(fold))*2*ratio, score)
        if best is None or best[3] < 10.0:
            return None
        n_id2, off, p0, _ = best
        Pf = self.period*fs
        y = x*phy.lte_phy.ramp(len(x), 0.0, off/fs)
        tmpl = pss_time(n_id2, nfft)
        blocks = []
        k = 0
        while p0 + k*Pf + nfft + 4*(nfft + 200) < len(y):
            r = self._find(y, int(round(p0 + k*Pf)), tmpl, ratio + 4)
            if r is not None and r[1] >= self.gate:
                m = self._measure(y, r[0], nfft, fs, n_id2)
                if m is not None:
                    blocks.append((r[0], m))
            k += 1
        if len(blocks) < 2:
            return None
        ids = [m[0] for _, m in blocks]
        n_id1 = max(set(ids), key=ids.count)
        blocks = [(p, m) for p, m in blocks if m[0] == n_id1]
        if len(blocks) < 2:
            return None
        pci = 3*n_id1 + n_id2
        pos = np.array([p for p, _ in blocks], float)
        raw = np.mean([self._bias(m[1]) for _, m in blocks])
        raw = raw/abs(raw) if abs(raw) > 0 else 1.0 + 0j
        bias, applied = self._snap(raw, blocks[0][1][1][2])
        dfs = np.array([self._offset(m[1], bias) for _, m in blocks])
        k = np.round((pos - pos[0])/Pf)
        scale = float(np.polyfit(k, pos, 1)[0]/Pf) if len(set(k)) > 1 else 1.0
        tt = pos/fs
        f0 = off + float(np.mean(dfs)) # (each block's offset, the mean at its mean time)
        rate = float(np.polyfit(tt, dfs, 1)[0]) if len(blocks) >= 5 else 0.0 # (a few blocks: noise; the loop finds it)
        p_last, m_last = blocks[-1]
        ibar, _ = self._ibar(m_last[3], m_last[4], pci)
        i_ssb, n_hf = ibar & 3, ibar >> 2
        S = phy.slot_len(nfft)
        l0 = phy.SSB_SYMBOLS[i_ssb % 2]
        # The frame start of the first block's frame: back from its PSS through its symbol, slot and half frame
        p_first = pos[0]
        frame0 = p_first - phy.symbol_starts(nfft)[l0]*scale - (5*n_hf + i_ssb//2)*S*scale
        tm = float(np.mean(tt))
        return SimpleNamespace(pci=pci, n_id1=n_id1, n_id2=n_id2, i_ssb=i_ssb, n_hf=n_hf, l0=l0, tmpl=tmpl, T=float(frame0),
                               scale=scale, f=f0 + rate*(frame0/fs - tm), rate=rate, phase=0.0, slot=0, miss=0, metric=0.0,
                               bias=bias, raw=raw, applied=applied)

    # ---- the stream

    def step(self, x, st, meta):
        fs = meta.Fs
        nfft = int(round(fs/phy.SCS))
        if nfft not in (128, 256, 512, 1024, 1536, 2048) or abs(nfft*phy.SCS - fs) > 1:
            raise ValueError(f"NrCell: {fs/1e6:g} MS/s is not 15 kHz times an FFT of 128 to 2048")
        x = np.asarray(x).astype(np.complex64)
        if st.buf is None:
            st.buf, st.base = x[:0], meta.pos
        st.buf = np.concatenate((st.buf, x))
        S = phy.slot_len(nfft)
        out = []
        while True:
            if st.lock is None:
                n = int(self.acquire*fs)
                if len(st.buf) < n and not meta.flush:
                    break
                if len(st.buf) < 2*self.period*fs:
                    break
                lock = self._acquire(st.buf[:n].astype(np.complex128), fs, nfft)
                if lock is None:
                    keep = int(self.period*fs)
                    drop = max(len(st.buf) - keep, 0) if len(st.buf) >= n else 0
                    st.buf, st.base = st.buf[drop:], st.base + drop
                    if len(st.buf) < n:
                        break
                    continue
                lock.T += st.base
                if st.est.get("next") is not None: # Acquired again: zeros for what was missed, a whole number of periods
                    per = int(round(self.period/1e-3))
                    gap = max(int(round((lock.T - st.est["next"])/S/per))*per, 0)
                    out.append(np.zeros(gap*S, np.complex64))
                    st.out_slots += gap
                while lock.T < st.base: # (a frame start before the buffer: the next frame that holds a block)
                    lock.T += 2*10*S*lock.scale
                    lock.f += lock.rate*2*10*S*lock.scale/fs
                st.lock = lock
                st.est.update(locked=True, acquired=st.est["acquired"] + 1, pci=lock.pci, n_id1=lock.n_id1, n_id2=lock.n_id2,
                              i_ssb=lock.i_ssb, n_hf=lock.n_hf)
            lk = st.lock
            a = int(round(lk.T)) - st.base
            need = a + S + 16
            if len(st.buf) < need:
                break
            # The slot derotated: the offset at its middle, linear across it (the rate's square term within a slot is below
            # 3e-4 cycles at 1 kHz/s), the phase continuous from slot to slot
            f_mid = lk.f + lk.rate*S/(2*fs)
            seg = (st.buf[a:a + S]*phy.lte_phy.ramp(S, lk.phase, f_mid/fs)).astype(np.complex128)
            slot_in_frame = lk.slot % 10
            frame = lk.slot//10
            per = int(round(self.period/1e-3)) # Slots a period
            if frame % (per//10) == 0 and slot_in_frame == 5*lk.n_hf + lk.i_ssb//2: # This slot holds the block: measure
                cp = phy.cp_lengths(nfft)[1]
                p = int(phy.symbol_starts(nfft)[lk.l0])
                r = self._find(seg, p, lk.tmpl, 6)
                m = self._measure(seg, r[0], nfft, fs, lk.n_id2, lk.n_id1) if r is not None else None
                if r is not None and r[1] >= self.gate and m is not None and m[2] > 0.3:
                    e = r[0] - p # Samples late
                    Tp = per*S # (samples between measurements)
                    lk.T += 0.5*e
                    lk.scale += 0.15*e/Tp
                    b = lk.raw*0.9 + 0.1*self._bias(m[1])
                    lk.raw = b/abs(b) if abs(b) > 0 else lk.raw
                    lk.bias, lk.applied = self._snap(lk.raw, m[1][2])
                    ef = self._offset(m[1], lk.bias)
                    lk.f += 0.5*ef
                    lk.rate += 0.15*ef/self.period
                    lk.miss, lk.metric = 0, r[1]
                    del cp
                else:
                    lk.miss += 1
            out.append(seg.astype(np.complex64))
            st.out_slots += 1
            L = S*lk.scale
            dt = L/fs
            lk.phase = (lk.phase + (lk.f + lk.rate*S/(2*fs))*(round(lk.T + L) - round(lk.T))/fs) % 1.0
            lk.f += lk.rate*dt
            lk.T += L
            lk.slot += 1
            st.est.update(slots=st.est["slots"] + 1, cfo=lk.f, rate=lk.rate, ppm=(lk.scale - 1)*1e6, metric=lk.metric,
                          next=lk.T, phase=lk.applied)
            drop = max(int(lk.T) - st.base - 64, 0)
            st.buf, st.base = st.buf[drop:], st.base + drop
            if lk.miss >= self.lose:
                st.lock = None
                st.est.update(locked=False, lost=st.est["lost"] + 1)
        meta.stream.est[self.name] = {k: v for k, v in st.est.items() if k != "next"}
        return np.concatenate(out) if out else x[:0]
