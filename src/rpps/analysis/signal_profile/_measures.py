"""The measures that decide which of the chain's steps to take (rpps.analysis.signal_profile)"""
import numpy as np

RATE_WIDTH = 0.85 # A signal narrower than this times its symbol rate wasn't timed at its rate
STRONG = 0.1      # A code re-encoding under this times a wrong code's mismatch confirms the rate it was found at
ESN0_GAP = 1.5 # dB: a linear carrier, locked, measures the Es/N0 its band's SNR gives to within about 1 dB (carriers on air
               # that decode cleanly come in 0.5-1 dB under; this leaves the band estimate a little room). Further under it,
               # the symbols don't sit on this constellation: not linear, a wrong rate, or carriers on top of each other

def esn0_gap(sig, det):
    """(expected Es/N0, gap) in dB: what the survey's in-band SNR (over the noise beside the signal) gives at the acquired
    rate (Signal.esn0_db), and how far the measured one falls under it; (None, None) when either is unknown"""
    if det.esn0 is None or getattr(sig, "snr", None) is None or not getattr(sig, "width", None):
        return None, None
    want = sig.esn0_db(det.baud)
    return want, want - det.esn0

def _strong(fec):
    """Whether a found code is evidence enough for the symbol rate too: one with more than a re-encoding (RS, HDLC, a
    block code, a matching shipped profile), or a convolutional code fitting far better than a wrong one"""
    if fec.kind != "conv" or fec.profile_name or fec.inner:
        return True
    return fec.mismatch is not None and fec.baseline and fec.mismatch < STRONG*fec.baseline

NEAR = 2**20 # Channel samples kept for rpps.analysis.spectrum.waveform
PM_OFFSET = 0.25 # A residual carrier this far (of the signal's width) from its centre isn't one: PM is symmetric about it
CARRIER_SHARE = 0.01 # Power share of the strongest spectral bin (of 8192) worth a residual-carrier analysis

def carrier_share(x, nfft=8192):
    """The strongest bin's share of the power in an averaged spectrum: a carrier line holds far more than noise-like
    modulation's 1/nfft or so (a quick screen before rpps.analysis.framing.telemetry)"""
    x = np.asarray(x)
    k = len(x)//nfft
    if k < 1:
        return 0.0
    P = np.mean(np.abs(np.fft.fft(x[:k*nfft].reshape(k, nfft), axis=1))**2, axis=0)
    return float(P.max()/P.sum())
SYMS = 2**18 # Tracked symbols kept for rpps.analysis.framing.frames

def _waveform_detail(w):
    d = {"envelope": w.envelope}
    if w.tones is not None:
        d["tones"] = [[f, sh] for f, sh in w.tones.lines]
    if w.ofdm is not None:
        d["ofdm"] = {"tu": w.ofdm.tu, "tcp": w.ofdm.tcp, "strength": w.ofdm.strength, "systems": [n for n, _ in w.ofdm.systems[:3]]}
    if w.bursts is not None:
        d["bursts"] = {"duty": w.bursts.duty, "period": w.bursts.period}
    return d
