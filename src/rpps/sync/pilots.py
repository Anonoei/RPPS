from types import SimpleNamespace

import numpy as np

from ..base import Stage

class PilotEqualizer(Stage):
    """Equalize OFDM carriers from their pilots: n_fft carrier values per OFDM symbol in, the same with the channel removed

    The channel seen on the pilots of each OFDM symbol is fitted with a short impulse response (least squares, `taps`
    long), and every carrier is divided by that response's frequency response. A multipath channel shorter than the prefix
    and a constant phase offset come off. Block estimation, one OFDM symbol at a time.

    Which OFDM symbol a block starts on (the pilots' sign alternates) comes from its absolute stream position, so any
    block cut gives the same output.

    carriers: the `rp.symbol.Carriers` layout. taps: channel length (default 3/4 of the pilots, at most `max_taps`)
    """
    def __init__(self, carriers, taps=None, max_taps=16):
        self.carriers = carriers
        k = np.asarray(carriers.pilots)
        self.order = np.argsort(k)
        self.pk = k[self.order]
        self.pb = carriers.bins(self.pk)
        n = carriers.n_fft
        L = taps or max(1, min(max_taps, (3*len(self.pk))//4))
        lags = np.arange(L)
        fit = np.linalg.pinv(np.exp(-2j*np.pi*np.outer(self.pk, lags)/n)) # pilot response -> impulse response
        all_k = np.arange(n)
        self.response = np.exp(-2j*np.pi*np.outer(all_k, lags)/n) @ fit # pilot response -> response on every bin

    def new_state(self, meta):
        return SimpleNamespace(held=None, held_pos=None)

    def step(self, samples, st, meta):
        n = self.carriers.n_fft
        if st.held is None:
            st.held, st.held_pos = np.asarray(samples)[:0], meta.pos
        x = np.concatenate((st.held, samples))
        m = len(x)//n
        first = st.held_pos//n # The OFDM symbol number of the first whole one: from the stream position
        st.held, st.held_pos = x[m*n:], st.held_pos + m*n
        if not m:
            return x[:0].astype(np.complex64)
        grid = x[:m*n].reshape(m, n)
        known = self.carriers.pilot_values(first + np.arange(m))[:, self.order]
        h_p = grid[:, self.pb]/known
        h = (self.response @ h_p.T).T
        return (grid/np.where(np.abs(h) < 1e-9, 1e-9, h)).reshape(-1).astype(np.complex64)

class PilotPhase(Stage):
    """Carrier phase over each frame from its pilot blocks: a line through the phases they measure, taken off the frame

        payload = symbols + rp.sync.PilotPhase(slots=360)       # DVB-S2: blocks of 36 pilots after every 16 slots of 90

    Block estimation per frame, data-aided: the phase of each pilot block (the mean of the symbols times the conjugate pilot
    value) at its centre, unwrapped in order, and a least-squares line through them (and through `lead`: a position before
    the payload where the phase is known to be 0, the frame's preamble once a frame sync took its phase off). A frame is held
    until it is whole. Without pilots, or before a profile says how the frame is laid out (`slots`, `pilots`), the symbols
    pass unchanged. Frames start where a profile applies a change (rpps.profile.control), or at the start of the stream.
    Estimates in stream.est[name]: the last frame's phase line (phase at the start, per symbol).
    """
    CONTROLS = {"slots": "live", "pilots": "live"}

    def __init__(self, slot=90, every=16, length=36, value=(1 + 1j)/np.sqrt(2), slots=None, pilots=True, lead=None,
                 name="pilots"):
        self.slot, self.every, self.length, self.value = int(slot), int(every), int(length), complex(value)
        self.slots, self.pilots, self.lead, self.name = slots, bool(pilots), lead, name
        self._start = True

    def __str__(self):
        return f"PilotPhase({self.name})"

    def control(self, slots=None, pilots=None):
        if slots is not None:
            self.slots = int(slots)
        if pilots is not None:
            self.pilots = bool(pilots)
        self._start = True

    def pos_rate(self, decode):
        return 1.0

    def new_state(self, meta):
        return SimpleNamespace(held=np.zeros(0, np.complex128), i=0)

    def _fit(self, y):
        from ..symbol.frame import pilot_layout
        _, starts = pilot_layout(self.slots, self.pilots, self.slot, self.every, self.length)
        pos = [s + self.length/2 for s in starts]
        ph = [float(np.angle(np.mean(y[s:s + self.length]*np.conj(self.value)))) for s in starts]
        if self.lead is not None:
            pos, ph = [float(self.lead)] + pos, [0.0] + ph
        if not ph:
            return y, (0.0, 0.0)
        ph = np.unwrap(ph)
        line = np.polyfit(pos, ph, 1) if len(ph) > 1 else np.array([0.0, ph[0]])
        return y*np.exp(-1j*np.polyval(line, np.arange(len(y)))), (float(line[1]), float(line[0]))

    def step(self, x, st, meta):
        out = []
        if self._start:
            from ..profile.control import anchor
            self._start = False
            if len(st.held): # A frame cut short: out as it came
                out.append(st.held)
            anchor(meta, self, meta.pos, len(st.held))
            st.held, st.i = np.zeros(0, np.complex128), 0
        x = np.asarray(x, dtype=np.complex128)
        if self.slots is None or not self.pilots:
            out.append(x)
        else:
            from ..symbol.frame import pilot_layout
            total, _ = pilot_layout(self.slots, self.pilots, self.slot, self.every, self.length)
            need = total - st.i
            if need > 0:
                st.held = np.concatenate((st.held, x[:need]))
                st.i += min(len(x), need)
                if st.i >= total:
                    y, line = self._fit(st.held)
                    out.append(y)
                    st.held = np.zeros(0, np.complex128)
                    meta.stream.est[self.name] = {"phase": line[0], "rate": line[1]}
                elif meta.flush:
                    out.append(st.held)
                    st.held = np.zeros(0, np.complex128)
            if len(x) > max(need, 0): # Past the frame: out unchanged (the pilot remover drops it)
                out.append(x[max(need, 0):])
                st.i += len(x) - max(need, 0)
        return np.concatenate(out).astype(np.complex64) if out else np.zeros(0, np.complex64)
