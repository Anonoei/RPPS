"""Project 25 full-rate IMBE vocoder, decoder side, written from TIA-102.BABA ("Project 25 Vocoder Description", 2003)

    from imbe import Decoder
    dec = Decoder()
    pcm = dec.decode(frame_bits)        # 144 bits (0/1) of one voice frame, as received -> 160 samples (8 kS/s, floats in 16-bit units)

What it does, section by section of the standard:
  7.5  de-interleave the 144 bits into the code vectors c0..c7 (Annex H)
  7.3 / 7.4  Golay [23,12] decode c0 (u0), generate the pseudo-random modulation from u0, demodulate c1..c7, Golay decode c1-c3, Hamming [15,11] decode c4-c6
  7.1  undo the bit prioritization: u0..u7 -> quantizer values b0..b(L+1) (the scan order of Figure 22, checked against the worked example of Section 10)
  6.1, 6.2, 6.4  pitch, voiced / unvoiced decisions, gain vector and DCT coefficients, log-spectral amplitudes predicted from the previous frame
  7.6, 7.7, 7.8  error estimate, frame repeat, muting
  8, 9  spectral amplitude enhancement, adaptive smoothing of the V/UV decisions and amplitudes
  11   unvoiced synthesis (windowed noise through a 256-point DFT, weighted overlap-add) and voiced synthesis (a sinusoid per harmonic with phase tracking)

Verification is limited: there is no reference decoder or reference audio here. The tables are checked for consistency (imbe_tables.py), the prioritization against the spec's own example, and the output
for plausibility (pitch track, voicing, level, spectrum); the speech quality itself has not been compared with another decoder. Meant to be upstreamed into RPPS.
"""
import os
import sys

import numpy as np

import rpps as rp

import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); import _paths  # noqa: E402,F401
import imbe_tables as T

N = 160 # samples per frame (20 ms at 8 kS/s)
WS = np.zeros(1000); WR = np.zeros(1000)
WS_OFF = 500


def ws(n):
    """Synthesis window w_S(n), zero outside -105..105 (n an integer array)"""
    n = np.asarray(n); out = np.zeros(n.shape)
    ok = (n >= -105) & (n <= 105)
    out[ok] = np.asarray(T.WS)[n[ok] + 105]
    return out


GAMMA_W = float(np.sum(T.WR)*np.sqrt(np.sum(np.square(T.WS))/np.sum(np.square(T.WR)))) # Eq. (121)

# ---------------------------------------------------------------------------------------------------- error control codes
# Golay (23,12) for u0-u3 and Hamming (15,11) for u4-u6 (TIA-102.BABA), from the RPPS definitions; integers MSB first
_GOLAY = rp.code.load("block", "golay/23_12")
_HAMMING = rp.code.load("block", "hamming/imbe_15_11", decoder="syndrome")


def _int_bits(v, n):
    return np.array([(v >> (n - 1 - i)) & 1 for i in range(n)], np.uint8)


def _bits_int(b):
    v = 0
    for x in b:
        v = (v << 1) | int(x)
    return v


def golay_encode(u):
    """23-bit codeword of 12 data bits: the data followed by 11 parity bits"""
    return _bits_int(_GOLAY._enc.encode(_int_bits(u, 12)))


def golay_decode(c):
    """(12 data bits, errors corrected) of a 23-bit word; errors = 4 when the word is not within 3 of a codeword"""
    u = _bits_int(_GOLAY._dec.decode(_int_bits(c, 23)))
    e = int(_GOLAY.errors[0])
    return u, (4 if e < 0 else e)


def hamming_decode(c):
    """(11 data bits, errors corrected) of a 15-bit word (data in bits 14..4, parity in bits 3..0)"""
    u = _bits_int(_HAMMING._dec.decode(_int_bits(c, 15)))
    return u, int(_HAMMING.errors[0])


# ---------------------------------------------------------------------------------------------------- frame layout and bit manipulation
LEN = [23, 23, 23, 23, 15, 15, 15, 7]


def deinterleave(bits):
    """144 received bits -> code vectors c0..c7 as integers (bit N-1 is the MSB), Annex H"""
    c = [0]*8
    for s, (a, na, b, nb) in enumerate(T.FRAME_MAP):
        c[a] |= int(bits[2*s]) << na
        c[b] |= int(bits[2*s + 1]) << nb
    return c


def pn_vectors(u0):
    """The modulation vectors m1..m7 as integers, Eqs. (84)-(93)"""
    pr = 16*u0; seq = []
    for _ in range(114):
        pr = (173*pr + 13849) % 65536
        seq.append(pr >> 15)
    m = [0]
    for lo, n in ((0, 23), (23, 23), (46, 23), (69, 15), (84, 15), (99, 15)):
        v = 0
        for k in range(n):
            v = (v << 1) | seq[lo + k]
        m.append(v)
    m.append(0)
    return m


def alloc(L):
    """Bits B_m of b3..b(L+1) as {m: B}: Annex F for m = 3..7, Annex G for m = 8..L+1"""
    d = {m: T.GAIN_ALLOC[L][m - 3][0] for m in range(3, 8)}
    for i, B in enumerate(T.DCT_ALLOC[L]):
        d[8 + i] = B
    return d


def scan_order(L):
    """The order in which the spectral quantizer bits (b3..b(L+1)) are scanned into the bit vectors: rows from the most significant bit down, left to right (Figure 22): list of (m, bit)"""
    B = alloc(L); out = []
    for r in range(max(B.values()), 0, -1):
        for m in range(3, L + 2):
            if B[m] >= r:
                out.append((m, r - 1))
    return out


def layout(K):
    """Where the quantizer bits go in the bit vectors (Section 7.1). u4, u5, u6 and bits 6-4 of u7 form one stream, filled in this order: the K bits of b1 (MSB first), bits 2 and 1 of b2, then the scanned
    spectral bits. (For K <= 9 b1 sits inside u4; for longer K it runs on into u5.) The head of the scan is u0 bits 2-0 and all of u1-u3. Returns (b1 slots, b2 slots, scan slots) as (vector, bit) lists."""
    head = [(0, 2), (0, 1), (0, 0)]
    for v in (1, 2, 3):
        head += [(v, b) for b in range(11, -1, -1)]
    tail = [(4, b) for b in range(10, -1, -1)] + [(5, b) for b in range(10, -1, -1)] + [(6, b) for b in range(10, -1, -1)] + [(7, 6), (7, 5), (7, 4)]
    return tail[:K], tail[K:K + 2], head + tail[K + 2:]


def block_lengths(L):
    base, rem = divmod(L, 6)
    return [base + (1 if i > 6 - rem else 0) for i in range(1, 7)] # Annex J: lengths differ by at most one and do not decrease


def unpack(u, L, K):
    """Bit vectors u0..u7 -> quantizer values {m: b_m} for m = 0..L+2 (inverse of Section 7.1)"""
    b = {m: 0 for m in range(L + 3)}
    b[0] = (((u[0] >> 6) & 0x3F) << 2) | (((u[7] >> 2) & 1) << 1) | ((u[7] >> 1) & 1)
    b[2] = (((u[0] >> 3) & 7) << 3) | ((u[7] >> 3) & 1)
    b1s, b2s, scan = layout(K)
    for k, (v, vb) in enumerate(b1s):
        b[1] |= ((u[v] >> vb) & 1) << (K - 1 - k)
    for k, (v, vb) in enumerate(b2s):
        b[2] |= ((u[v] >> vb) & 1) << (2 - k)
    b[L + 2] = u[7] & 1
    for (m, bit), (v, vb) in zip(scan_order(L), scan):
        b[m] |= ((u[v] >> vb) & 1) << bit
    return b


# ---------------------------------------------------------------------------------------------------- the decoder
STEP_BY_BITS = {1: 1.2, 2: 0.85, 3: 0.65, 4: 0.40, 5: 0.28, 6: 0.15, 7: 0.08, 8: 0.04, 9: 0.02, 10: 0.01} # Table 3
SIGMA = {2: 0.307, 3: 0.241, 4: 0.207, 5: 0.190, 6: 0.179, 7: 0.173, 8: 0.165, 9: 0.170, 10: 0.170} # Table 4


def params(b, L, K):
    """(omega0, v[1..L] bool, T[1..L], ...) from the quantizer values; returns (log2-domain prediction residual T_l (array, l = 1..L), v)"""
    B = alloc(L)
    # V/UV decisions (Eqs. 50-51)
    v = np.zeros(L + 1, bool)
    for l in range(1, L + 1):
        lam = (l + 2)//3 if l <= 36 else 12
        v[l] = ((b[1] >> (K - lam)) & 1) == 1
    # gain vector (Annex E, F; Eqs. 68-70)
    G = np.zeros(7)
    G[1] = T.GAIN_LEVELS[b[2]]
    for m in range(3, 8):
        Bm, step = T.GAIN_ALLOC[L][m - 3]
        G[m - 1] = 0.0 if Bm == 0 else step*(b[m] - 2**(Bm - 1) + 0.5)
    R = np.zeros(7)
    for i in range(1, 7):
        R[i] = sum((1 if m == 1 else 2)*G[m]*np.cos(np.pi*(m - 1)*(i - 0.5)/6) for m in range(1, 7))
    # higher order DCT coefficients (Annex G, Tables 3 and 4; Eqs. 71-74)
    J = block_lengths(L); Tl = []; m = 8
    for i in range(1, 7):
        C = np.zeros(J[i - 1] + 1)
        C[1] = R[i]
        for k in range(2, J[i - 1] + 1):
            Bm = B[m]
            C[k] = 0.0 if Bm == 0 else STEP_BY_BITS[Bm]*SIGMA[k]*(b[m] - 2**(Bm - 1) + 0.5)
            m += 1
        for j in range(1, J[i - 1] + 1):
            Tl.append(sum((1 if k == 1 else 2)*C[k]*np.cos(np.pi*(k - 1)*(j - 0.5)/J[i - 1]) for k in range(1, J[i - 1] + 1)))
    return np.array(Tl), v


class Decoder:
    def __init__(self):
        self.w0 = 0.02985; self.L = 30; self.K = 10
        self.v = np.zeros(57, bool)               # smoothed V/UV of the previous frame
        self.lm = np.zeros(58)                    # log2 of the previous frame's unenhanced amplitudes, index 0 = 1.0 (log2 = 0)
        self.Lm = 30                              # the L the amplitudes belong to
        self.M = np.zeros(57)                     # enhanced, smoothed amplitudes of the previous frame (used by the synthesis)
        self.eR = 0.0; self.SE = 75000.0; self.tauM = 20480.0
        self.psi = np.zeros(57); self.phi = np.zeros(57)
        self.uw = np.zeros(256)                   # previous frame's unvoiced signal, index n mod 256
        self.noise = [3147]                       # u(-105), ...
        self.frames = 0; self.prev_w0 = 0.02985; self.prev_L = 30
        self.errors = 0; self.repeats = 0; self.mutes = 0
        self.last = None

    # noise sequence u(n), Eq. (117): frame t uses x[160 t + n + 105]
    def _noise(self, t):
        need = 160*t + 211
        while len(self.noise) < need:
            x = self.noise[-1]
            self.noise.append((171*x + 11213) % 53125)
        return np.array(self.noise[160*t:160*t + 211], dtype=float)  # u(-105..105)

    def decode(self, bits):
        """144 received bits -> 160 samples. Sets self.last to a dict of the frame's parameters and error counts"""
        c = deinterleave(bits)
        u0, e0 = golay_decode(c[0])
        m = pn_vectors(u0)
        u = [u0]; eps = [e0]
        for i in (1, 2, 3):
            ui, ei = golay_decode(c[i] ^ m[i]); u.append(ui); eps.append(ei)
        for i in (4, 5, 6):
            ui, ei = hamming_decode(c[i] ^ m[i]); u.append(ui); eps.append(ei)
        u.append(c[7] ^ m[7])
        Tt = sum(eps)
        eR = 0.95*self.eR + 0.000365*Tt            # Eq. (96)
        b0 = (((u[0] >> 6) & 0x3F) << 2) | (((u[7] >> 2) & 1) << 1) | ((u[7] >> 1) & 1)
        invalid = b0 > 207
        if not invalid:
            w0 = 4*np.pi/(b0 + 39.5)                 # Eq. (46)
            L = int(0.9254*int(np.pi/w0 + 0.25))     # Eq. (47)
            invalid = not (9 <= L <= 56)
        repeat = invalid or (e0 >= 2 and Tt >= 10 + 40*eR)  # Eqs. (97)-(98)
        self.eR = eR
        info = dict(b0=b0, eps=eps, T=Tt, eR=eR, repeat=repeat, mute=False)
        if repeat:
            self.repeats += 1
            w0, L, K = self.prev_w0, self.prev_L, self.K
            v = self.v.copy(); Mbar = self.M.copy()
            vt = v
            if eR > 0.0875:
                info["mute"] = True; self.mutes += 1
                self.last = info
                return self._noise_frame()
        else:
            K = (L + 2)//3 if L <= 36 else 12
            b = unpack(u, L, K)
            Tl, vt = params(b, L, K)
            rho = 0.4 if L <= 15 else (0.03*L - 0.05 if L <= 24 else 0.7)   # Eq. (55)
            Lp = self.Lm
            lmp = np.concatenate([self.lm[:Lp + 1], np.full(L + 2, self.lm[Lp])]) # Eq. (79): beyond L(-1) the last amplitude repeats; index 0 is M_0 = 1
            ll = np.arange(1, L + 1)
            kl = Lp/L*ll; fl = np.floor(kl).astype(int); dl = kl - fl
            pred = (1 - dl)*lmp[fl] + dl*lmp[fl + 1]
            lm = Tl + rho*pred - rho/L*np.sum(pred)  # Eq. (77), log2 amplitudes, l = 1..L
            Mt = np.concatenate([[0.0], 2.0**lm])
            # enhancement, Eqs. (105)-(110)
            R0 = np.sum(Mt[1:]**2); R1 = np.sum(Mt[1:]**2*np.cos(w0*ll))
            Wl = np.sqrt(Mt[1:])*(0.96*np.pi*(R0**2 + R1**2 - 2*R0*R1*np.cos(w0*ll))/(w0*R0*(R0**2 - R1**2)))**0.25
            Mb = np.where(8*ll <= L, Mt[1:], np.where(Wl > 1.2, 1.2*Mt[1:], np.where(Wl < 0.5, 0.5*Mt[1:], Wl*Mt[1:])))
            Mb = Mb*np.sqrt(R0/np.sum(Mb**2))
            SE = 0.95*self.SE + 0.05*R0
            self.SE = SE if SE <= 10000.0 else 10000.0   # Eq. (111)
            # smoothing of the V/UV decisions, Eqs. (112)-(113)
            if eR <= 0.005 and Tt <= 4:
                VM = np.inf
            elif eR <= 0.0125 and eps[0] == 0:
                VM = 45.255*self.SE**0.375/np.exp(277.26*eR)
            else:
                VM = 1.414*self.SE**0.375
            v = np.zeros(L + 1, bool); v[1:] = vt[1:] | (Mb > VM)
            # smoothing of the amplitudes, Eqs. (114)-(116)
            AM = np.sum(Mb)
            self.tauM = 20480.0 if (eR <= 0.005 and Tt <= 6) else 6000.0 - 300.0*Tt + self.tauM
            Mb = Mb*(1.0 if self.tauM > AM else self.tauM/AM)
            Mbar = np.concatenate([[0.0], Mb])
            # remember for the next frame's prediction (unenhanced, log domain)
            self.lm = np.concatenate([[0.0], lm]); self.Lm = L
        self.K = K
        info.update(w0=w0, L=L, K=K, voiced=int(np.sum(v[1:])))
        pcm = self._synth(w0, L, v, Mbar)
        self.prev_w0 = w0; self.prev_L = L
        self.v = np.zeros(57, bool); self.v[:L + 1] = v[:L + 1]
        self.M = np.zeros(57); self.M[:L + 1] = Mbar[:L + 1]
        self.frames += 1
        self.last = info
        return pcm

    def _noise_frame(self):
        self.frames += 1
        return np.random.default_rng(self.frames).uniform(-5, 5, N)

    def _synth(self, w0, L, v, M):
        t = self.frames
        u = self._noise(t) # u(-105..105)
        n = np.arange(N)
        w0p, Lp = self.prev_w0, self.prev_L
        Lmax = max(L, Lp)
        # unvoiced component, Section 11.2
        arr = np.zeros(256)
        nn = np.arange(-104, 105)
        arr[nn % 256] = u[nn + 105]*ws(nn)
        U = np.fft.fft(arr)            # U_w(m) for m mod 256
        Ut = np.zeros(256, complex)
        for l in range(1, L + 1):
            if v[l]:
                continue
            lo = int(np.ceil(256/(2*np.pi)*(l - 0.5)*w0)); hi = int(np.ceil(256/(2*np.pi)*(l + 0.5)*w0))
            hi = min(hi, 128)
            if hi <= lo:
                continue
            band = np.sqrt(np.sum(np.abs(U[lo:hi])**2)/(hi - lo))
            if band <= 0:
                continue
            g = GAMMA_W*M[l]/band
            for mm in range(lo, hi):
                Ut[mm] = g*U[mm]
                Ut[(-mm) % 256] = g*U[(-mm) % 256]
        uw = np.fft.ifft(Ut).real     # index n mod 256, n = -128..127
        idx = lambda a: a % 256
        nm = n - N
        t1 = ws(n)*np.where(n <= 127, self.uw[idx(n)], 0.0)
        t2 = ws(nm)*np.where(nm >= -128, uw[idx(nm)], 0.0)
        den = ws(n)**2 + ws(nm)**2
        s_uv = np.where(den > 0, (t1 + t2)/np.where(den > 0, den, 1.0), 0.0)
        # phases, Eqs. (139)-(141)
        Luv = int(np.sum(~v[1:L + 1]))
        psi = self.psi.copy(); phi = np.zeros(57)
        for l in range(1, 57):
            psi[l] = self.psi[l] + (w0p + w0)*l*N/2
        for l in range(1, 57):
            if l <= L//4 or l > Lmax:
                phi[l] = psi[l]
            else:
                rho = 2*np.pi*u[l + 105]/53125 - np.pi
                phi[l] = psi[l] + Luv*rho/L
        # voiced component, Section 11.3
        s_v = np.zeros(N)
        wn = ws(n); wnN = ws(n - N)
        for l in range(1, Lmax + 1):
            vp = bool(self.v[l]) if l <= Lp else False
            vc = bool(v[l]) if l <= L else False
            Mp = self.M[l] if l <= Lp else 0.0
            Mc = M[l] if l <= L else 0.0
            if not vp and not vc:
                continue
            if vp and not vc:
                s_v += 2*wn*Mp*np.cos(w0p*n*l + self.phi[l])
            elif vc and not vp:
                s_v += 2*wnN*Mc*np.cos(w0*(n - N)*l + phi[l])
            elif l >= 8 or abs(w0 - w0p) >= 0.1*w0:
                s_v += 2*(wn*Mp*np.cos(w0p*n*l + self.phi[l]) + wnN*Mc*np.cos(w0*(n - N)*l + phi[l]))
            else:
                dphi = phi[l] - self.phi[l] - (w0p + w0)*l*N/2
                dw = (dphi - 2*np.pi*np.floor((dphi + np.pi)/(2*np.pi)))/N
                th = self.phi[l] + (w0p*l + dw)*n + (w0 - w0p)*l*n**2/(2*N)
                a = Mp + n/N*(Mc - Mp)
                s_v += 2*a*np.cos(th)
        self.psi = psi; self.phi = phi; self.uw = uw
        return s_uv + s_v
