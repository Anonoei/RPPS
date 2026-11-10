"""P25 half-rate vocoder (the phase 2 voice codec, TIA-102.BABA-1): 72 channel bits -> 160 samples of 8 kS/s speech

    import p25p2_voice
    dec = p25p2_voice.HalfRate()
    pcm = dec.decode([c0, c1, c2, c3])      # code vectors as lists of bits, most significant first (p25p2.voice_codewords)

Decoding follows Clause 4 and 5 of the addendum: c0 is the [24,12] extended Golay code, c1 the [23,12] Golay code modulated by a pseudo-random sequence keyed on u0, c2 and c3 are uncoded;
the 49 bits of u0..u3 are put back into the nine quantizer values b0..b8 (Tables 5-8); b0 gives the pitch and the number of harmonics L (Annex A), b1 the voicing (Annex B), b2 the gain, b3/b4 the PRBA
vector, b5..b8 the higher order DCT coefficients; the log spectral amplitudes are predicted from the previous frame (Eq. 40-43). The voiced / unvoiced synthesis, the spectral amplitude
enhancement and the smoothing are the ones of the full-rate decoder (imbe.py, TIA-102.BABA chapters 8, 9 and 11), which the addendum refers to. Frame repeats and mutes follow Clause 5.5 to 5.7.
Not covered: tone frames (b0 = 126, 127) are treated as frame repeats.
UNTESTED against real speech: no clear (unencrypted) phase 2 call has been recorded yet, so only the structure (a bit-exact round trip of the quantizer values, and a plausibility check of the
synthesised signal) has been checked. Two readings of the draft addendum are noted in the code. Meant to be upstreamed into RPPS.
"""
import numpy as np

import rpps as rp

import imbe
import p25p2_tables as T


def _bits_to_int(bits):
    v = 0
    for b in bits:
        v = (v << 1) | int(b)
    return v


def pn_modulation(u0):
    """The 23-bit modulation vector m1 for the [23,12] code, keyed on u0 (Eq. 52-54); first element = most significant bit"""
    pr = 16*u0; m = 0
    for _ in range(23):
        pr = (173*pr + 13849) & 0xFFFF
        m = (m << 1) | (pr >> 15)
    return m


_GOLAY24 = rp.coding.load("block", "golay/24_12") # extended Golay, overall parity bit last


def golay24_decode(c0):
    """(12 data bits, errors) of the 24-bit extended Golay word (LSB = overall parity); errors = 4 when uncorrectable (4 errors are detected)"""
    u = imbe._bits_int(_GOLAY24._dec.decode(imbe._int_bits(c0, 24)))
    e = int(_GOLAY24.errors[0])
    return u, (4 if e < 0 else e)


def quantizers(u):
    """b0..b8 from u0..u3 (Tables 5-8): u = [u0 (12 bits), u1 (12), u2 (11), u3 (14)]"""
    u0, u1, u2, u3 = u
    b = [0]*9
    b[0] = (((u0 >> 8) & 0xF) << 3) | ((u3 >> 9) & 7)
    b[1] = (((u0 >> 4) & 0xF) << 1) | ((u3 >> 13) & 1)
    b[2] = ((u0 & 0xF) << 1) | ((u3 >> 12) & 1)
    b[3] = (((u1 >> 4) & 0xFF) << 1) | ((u3 >> 8) & 1)
    b[4] = ((u1 & 0xF) << 3) | ((u3 >> 5) & 7)
    b[5] = (((u2 >> 7) & 0xF) << 1) | ((u3 >> 4) & 1)
    b[6] = (((u2 >> 4) & 7) << 1) | ((u3 >> 3) & 1)
    b[7] = (((u2 >> 1) & 7) << 1) | ((u3 >> 2) & 1)
    b[8] = ((u2 & 1) << 2) | (u3 & 3)
    return b


def code_vectors(b):
    """The inverse of quantizers() and the FEC: b0..b8 -> [c0 (24 bits), c1 (23), c2 (11), c3 (14)] as lists of bits, MSB first (used for tests)"""
    u0 = (((b[0] >> 3) & 0xF) << 8) | (((b[1] >> 1) & 0xF) << 4) | ((b[2] >> 1) & 0xF)
    u1 = (((b[3] >> 1) & 0xFF) << 4) | ((b[4] >> 3) & 0xF)
    u2 = (((b[5] >> 1) & 0xF) << 7) | (((b[6] >> 1) & 7) << 4) | (((b[7] >> 1) & 7) << 1) | ((b[8] >> 2) & 1)
    u3 = ((b[1] & 1) << 13) | ((b[2] & 1) << 12) | ((b[0] & 7) << 9) | ((b[3] & 1) << 8) | ((b[4] & 7) << 5) | ((b[5] & 1) << 4) | ((b[6] & 1) << 3) | ((b[7] & 1) << 2) | (b[8] & 3)
    c0 = imbe._bits_int(_GOLAY24._enc.encode(imbe._int_bits(u0, 12)))
    c1 = imbe.golay_encode(u1) ^ pn_modulation(u0)
    f = lambda v, n: [(v >> (n - 1 - i)) & 1 for i in range(n)]
    return [f(c0, 24), f(c1, 23), f(u2, 11), f(u3, 14)]


def dct_inverse(C, J):
    """c_j = sum_k alpha(k) C_k cos(pi (k-1)(j-1/2)/J), j = 1..J (Eq. 38); C indexed from k = 1 (C[0] is k = 1)"""
    j = np.arange(1, J + 1)
    out = np.zeros(J)
    for k in range(1, J + 1):
        out += (1.0 if k == 1 else 2.0)*C[k - 1]*np.cos(np.pi*(k - 1)*(j - 0.5)/J)
    return out


class HalfRate(imbe.Decoder):
    def __init__(self):
        super().__init__()
        self.gamma = 0.0; self.lam = np.ones(58); self.Lm = 15      # gamma(-1) = 0, Lambda_l(-1) = 1, L(-1) = 15 (Clause 4.4)
        self.eR = 0.0; self.invalid_run = 0
        self.prev_w0 = 2*np.pi/32; self.prev_L = 14

    def parameters(self, b, w0, L, v_vec):
        """Log2 spectral amplitudes Lambda_l (l = 1..L), the prediction residuals and the gain, from b2..b8 (Clause 4.4)"""
        gamma = T.GAIN[b[2]] + 0.5*self.gamma                       # Eq. 26
        G = np.zeros(9); G[2:5] = T.PRBA24[b[3]]; G[5:9] = T.PRBA58[b[4]]
        i = np.arange(1, 9)
        R = np.array([sum((1.0 if m == 1 else 2.0)*G[m]*np.cos(np.pi*(m - 1)*(ii - 0.5)/8) for m in range(1, 9)) for ii in i])   # Eq. 27
        J = T.BLOCKS[L]
        C = [[0.0]*Ji for Ji in J]
        for blk in range(4):
            r1, r2 = R[2*blk], R[2*blk + 1]
            C[blk][0] = 0.5*(r1 + r2); C[blk][1] = (r1 - r2)/(2*np.sqrt(2))                          # Eq. 29-36
            H = T.HOC[blk][b[5 + blk]]
            for k in range(3, J[blk] + 1):
                if k - 2 <= 4:   # the printed Eq. 37 says "k <= 4"; the encoder text (Clause 4.3.2: H_{i,j} = C_{i,j+2}, 4-element vectors) needs k - 2 <= 4
                    C[blk][k - 1] = H[k - 3]
        Tl = np.concatenate([dct_inverse(C[blk], J[blk]) for blk in range(4)])                       # Eq. 38-39
        assert len(Tl) == L
        ll = np.arange(1, L + 1)
        Lp = self.Lm
        lmp = np.concatenate([[self.lam[1]], self.lam[1:Lp + 1], np.full(L + 2, self.lam[Lp])])       # Eq. 44-45
        kl = Lp/L*ll; fl = np.floor(kl).astype(int); dl = kl - fl                                       # Eq. 40-41
        pred = (1 - dl)*lmp[fl] + dl*lmp[fl + 1]
        Gam = gamma - 0.5*np.log2(L) - np.mean(Tl)                                                    # Eq. 42
        lam = Tl + 0.65*pred - 0.65/L*np.sum(pred) + Gam                                              # Eq. 43
        return gamma, lam

    def decode(self, c):
        """[c0, c1, c2, c3] (bit lists) -> 160 samples. self.last holds the frame's parameters and error counts"""
        c0 = _bits_to_int(c[0]); c1 = _bits_to_int(c[1])
        u0, e0 = golay24_decode(c0)
        u1, e1 = imbe.golay_decode(c1 ^ pn_modulation(u0))
        u = [u0, u1, _bits_to_int(c[2]), _bits_to_int(c[3])]
        eT = e0 + e1
        self.eR = 0.95*self.eR + 0.001064*eT                                                          # Eq. 55-56
        b = quantizers(u)
        kind = "voice" if b[0] < 120 else "erasure" if b[0] < 124 else "silence" if b[0] < 126 else "tone"
        repeat = kind in ("erasure", "tone") or e0 >= 4 or (e0 >= 2 and eT >= 6)                       # Clause 5.6
        info = dict(b=b, kind=kind, e0=e0, e1=e1, eT=eT, eR=self.eR, repeat=repeat, mute=False)
        if repeat:
            self.repeats += 1; self.invalid_run += 1
            if self.eR > 0.096 or self.invalid_run >= 4:                                              # Clause 5.7
                info["mute"] = True; self.mutes += 1; self.last = info
                return self._noise_frame()
            w0, L = self.prev_w0, self.prev_L
            v = self.v[:L + 1].copy(); Mbar = self.M[:L + 1].copy()
            return self._finish(info, w0, L, v, Mbar, update=False)
        self.invalid_run = 0
        if kind == "silence":
            w0, L = 2*np.pi/32, 14; vv = np.zeros(L + 1, bool)                                         # Eq. 1-3
        else:
            L, f0 = T.PITCH[b[0]]; w0 = 2*np.pi*f0   # Annex A lists w0 / 2 pi
            vec = T.VUV[b[1]]
            ll = np.arange(1, L + 1); jl = np.minimum(7, np.floor(16*w0*ll/(2*np.pi)).astype(int))     # Eq. 5, 7
            vv = np.zeros(L + 1, bool); vv[1:] = np.array(vec)[jl].astype(bool)
        gamma, lam = self.parameters(b, w0, L, None)
        self.gamma = gamma; self.lam = np.concatenate([[0.0], lam, np.zeros(57 - L)]); self.Lm = L
        ll = np.arange(1, L + 1)
        M = np.where(vv[1:], 2.0**lam, 0.2046/np.sqrt(w0)*2.0**lam)                                    # Eq. 46
        Mt = np.concatenate([[0.0], M])
        # spectral amplitude enhancement and smoothing, as the full-rate decoder (imbe.py)
        R0 = np.sum(Mt[1:]**2); R1 = np.sum(Mt[1:]**2*np.cos(w0*ll))
        den = (w0*R0*(R0**2 - R1**2))
        if den > 0 and R0 > 0:
            Wl = np.sqrt(Mt[1:])*(0.96*np.pi*(R0**2 + R1**2 - 2*R0*R1*np.cos(w0*ll))/den)**0.25
            Mb = np.where(8*ll <= L, Mt[1:], np.where(Wl > 1.2, 1.2*Mt[1:], np.where(Wl < 0.5, 0.5*Mt[1:], Wl*Mt[1:])))
            Mb = Mb*np.sqrt(R0/max(1e-12, np.sum(Mb**2)))
        else:
            Mb = Mt[1:].copy()
        SE = 0.95*self.SE + 0.05*R0; self.SE = min(SE, 10000.0)
        if self.eR <= 0.005 and eT <= 4:
            VM = np.inf
        elif self.eR <= 0.0125 and e0 == 0:
            VM = 45.255*self.SE**0.375/np.exp(277.26*self.eR)
        else:
            VM = 1.414*self.SE**0.375
        v = np.zeros(L + 1, bool); v[1:] = vv[1:] | (Mb > VM)
        AM = np.sum(Mb)
        self.tauM = 20480.0 if (self.eR <= 0.005 and eT <= 6) else 6000.0 - 300.0*eT + self.tauM
        Mb = Mb*(1.0 if self.tauM > AM else self.tauM/AM)
        return self._finish(info, w0, L, v, np.concatenate([[0.0], Mb]), update=True)

    def _finish(self, info, w0, L, v, Mbar, update):
        info.update(w0=w0, L=L, voiced=int(np.sum(v[1:])))
        pcm = self._synth(w0, L, v, Mbar)
        self.prev_w0 = w0; self.prev_L = L
        self.v = np.zeros(57, bool); self.v[:L + 1] = v[:L + 1]
        self.M = np.zeros(57); self.M[:L + 1] = Mbar[:L + 1]
        self.frames += 1
        self.last = info
        return pcm
