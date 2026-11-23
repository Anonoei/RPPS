"""SNR before lock: Zadoff-Chu sequences, from a known sequence (rpps.sync.quality.from_sequence), from the signal's
cyclostationarity (cyclic), from the band it leaves empty (blind) and from the symbols' moments (m2m4); and detect
scoring each candidate at its own moment Es/N0"""
import numpy as np
import pytest

import rpps as rp
from rpps import symbol
from rpps.grid.lte import cell
from rpps.sync import quality as Q

from conftest import shaped

def awgn(rng, n, var):
    return np.sqrt(var/2)*(rng.standard_normal(n) + 1j*rng.standard_normal(n))

def frontend(n, frac):
    """Noise through a front-end filter passing `frac` of Fs, 26 dB down outside (the signal's band untouched)"""
    N = np.fft.fft(n)
    N[np.abs(np.fft.fftfreq(len(n))) > frac/2] *= 0.05
    return np.fft.ifft(N)

def noisy(mod, esn0, sps, seed, beta=0.35, front=None, n_sym=2048):
    rng = np.random.default_rng(seed)
    x, _ = shaped(rng, symbol.load_or_generate(mod).points, n_sym, sps, beta=beta, tau=rng.random())
    n = awgn(rng, len(x), 10**(-esn0/10)) # per-sample variance 10^(-Es/N0/10): Es/N0 at unit symbol energy
    return x + (n if front is None else frontend(n, front))

# ---- Zadoff-Chu

def test_zadoff_chu_is_cazac():
    z = rp.sequence.zadoff_chu(129, 839)
    assert np.allclose(np.abs(z), 1)
    c = np.fft.ifft(np.fft.fft(z)*np.conj(np.fft.fft(z)))
    assert abs(c[0]) == pytest.approx(839) and np.max(np.abs(c[1:])) < 1e-6
    assert np.allclose(rp.sequence.zadoff_chu(129, 839, 13), np.roll(z, -13))

def test_lte_pss_is_the_punctured_root():
    for u in (25, 29, 34):
        n = np.arange(62)
        ref = np.where(n < 31, np.exp(-1j*np.pi*u*n*(n + 1)/63), np.exp(-1j*np.pi*u*(n + 1)*(n + 2)/63))
        assert np.allclose(cell.pss(u), ref)

@pytest.mark.parametrize("u, N", [(0, 63), (63, 63), (21, 63)])
def test_zadoff_chu_refuses_a_bad_root(u, N):
    with pytest.raises(ValueError):
        rp.sequence.zadoff_chu(u, N)

# ---- from a known sequence

@pytest.mark.parametrize("N, u, snr, bias, std", [(839, 129, -10, 0.4, 1.0), (839, 129, 0, 0.2, 0.45),
                                                   (839, 129, 20, 0.2, 0.3), (63, 25, 0, 0.6, 1.6)])
def test_from_sequence_zc(N, u, snr, bias, std):
    """Any cyclic shift and carrier phase; signal and noise from the one correlation (measured margins over 200 seeds:
    N=839 0.0/0.7, 0.0/0.3, 0.0/0.16; N=63 +0.1/1.1)"""
    z = rp.sequence.zadoff_chu(u, N)
    err = []
    for seed in range(60):
        rng = np.random.default_rng(seed)
        d = int(rng.integers(N))
        y = np.exp(2j*np.pi*rng.random())*np.roll(z, d) + awgn(rng, N, 10**(-snr/10))
        r = Q.from_sequence(y, z)
        assert r.delay == d
        err.append(r.snr_db - snr)
    assert abs(np.mean(err)) < bias and np.std(err) < std

def test_from_sequence_takes_the_channel_taps_and_coloured_noise():
    """Two taps (1 and 0.5 one sample later): the signal is both. Noise through a front-end filter: in band"""
    z = rp.sequence.zadoff_chu(129, 839)
    err = []
    for seed in range(30):
        rng = np.random.default_rng(seed)
        y = z + 0.5*np.roll(z, 1)
        y = y/np.sqrt(1.25) + frontend(awgn(rng, 839, 0.1), 1.0) # unit signal, 10 dB
        err.append(Q.from_sequence(y, z).snr_db - 10)
    assert abs(np.mean(err)) < 0.3

def test_from_sequence_least_squares():
    """Any known aligned sequence (not CAZAC): the gain fitted, the residual as the noise"""
    pts = symbol.load_or_generate("QPSK").points
    err = []
    for seed in range(40):
        rng = np.random.default_rng(seed)
        ref = pts[rng.integers(0, 4, 500)]
        h = 0.7*np.exp(1j)
        y = h*ref + awgn(rng, 500, abs(h)**2/10)
        r = Q.from_sequence(y, ref, periodic=False)
        assert abs(r.gain - h) < 0.05
        err.append(r.snr_db - 10)
    assert abs(np.mean(err)) < 0.2 and np.std(err) < 0.4

def test_from_sequence_needs_equal_lengths():
    with pytest.raises(ValueError):
        Q.from_sequence(np.ones(100), rp.sequence.zadoff_chu(25, 63))

# ---- from the signal's cyclostationarity

@pytest.mark.parametrize("mod", ["QPSK", "64QAM"])
@pytest.mark.parametrize("sps", [2.5, 4])
@pytest.mark.parametrize("beta", [0.2, 0.35, 0.5])
def test_cyclic(mod, sps, beta):
    """Es/N0 at 0 and 5 dB, the roll-off fitted (measured over 16 seeds: within -0.5 dB mean, 1.3 std; roll-off within
    0.07); front-end-filtered noise moves nothing on average, whatever the filter does outside the signal's band (one
    block's roll-off or rate may tip to the next step where the overlap is a few bins)"""
    for es in (0, 5):
        err, betas, moved = [], [], []
        for seed in range(12):
            r = Q.cyclic(noisy(mod, es, sps, seed, beta), sps)
            if r is None: # (roll-off 0.2 at 0 dB: now and then no correlation found)
                continue
            err.append(r.esn0_db - es)
            betas.append(r.beta)
            f = Q.cyclic(noisy(mod, es, sps, seed, beta, front=0.8), sps) # the band is at most 0.6 of Fs here
            if f is not None:
                moved.append(f.esn0_db - r.esn0_db)
        assert len(err) >= 10
        assert abs(np.mean(err)) < 1.0 and np.std(err) < 2.0
        assert abs(np.median(betas) - beta) < 0.08
        assert abs(np.mean(moved)) < 0.3

def test_cyclic_on_noise_alone_and_high_snr():
    rng = np.random.default_rng(0)
    quiet = 0
    for seed in range(60):
        r = Q.cyclic(awgn(np.random.default_rng(seed), 8192, 1.0), 4)
        quiet += r is None or r.esn0_db < -3
    assert quiet >= 54 # measured 97 of 100
    r = Q.cyclic(noisy("QPSK", 30, 4, 1), 4)
    assert r is not None and r.esn0_db > 10 # high: N0 is a small difference (or inf)
    assert Q.cyclic(rng.standard_normal(8192) + 0j, 1.0) is None # no band past the symbol rate's

# ---- from the empty band

@pytest.mark.parametrize("sps", [2.5, 4])
def test_blind_in_white_noise(sps):
    for es in (0, 10, 20):
        err = [Q.blind(noisy("QPSK", es, sps, seed), sps).esn0_db - es for seed in range(8)]
        assert abs(np.mean(err)) < 0.5

def test_blind_says_none_where_it_cant_tell():
    assert Q.blind(noisy("QPSK", 10, 4, 0), 1.0) is None # no empty band
    assert Q.blind(noisy("QPSK", -10, 4, 0), 4) is None # no signal found
    assert Q.blind(noisy("QPSK", 10, 4, 0, front=0.8), 4) is None # noise past the signal not flat: a filter's edge

# ---- from the symbols' moments

@pytest.mark.parametrize("mod, bias", [("QPSK", 0.3), ("8PSK", 0.3), ("16QAM", 0.5), ("32APSK", 0.5)])
def test_m2m4(mod, bias):
    """The right constellation's kurtosis gives the Es/N0 from 0 to 10 dB, any carrier phase, no decisions"""
    pts = symbol.load_or_generate(mod).points
    pts = pts/np.sqrt(np.mean(np.abs(pts)**2))
    for es in (0, 5, 10):
        err = []
        for seed in range(20):
            rng = np.random.default_rng(seed)
            y = pts[rng.integers(0, len(pts), 4096)]*np.exp(2j*np.pi*rng.random()) + awgn(rng, 4096, 10**(-es/10))
            err.append(Q.m2m4(y, pts) - es)
        assert abs(np.mean(err)) < bias

# ---- detect at each candidate's own Es/N0

NAMES = ("BPSK", "QPSK", "8PSK", "16PSK", "16QAM", "32QAM", "64QAM", "16APSK", "32APSK")

def _detect(mod, level, seed, **kw):
    x = noisy(mod, level, 4, seed)*np.exp(2j*np.pi*np.random.default_rng(seed + 99).random())
    return rp.sync.detect(x.astype(np.complex64), 1e6, names=NAMES, baud=0.25e6, **kw)

@pytest.mark.slow
def test_detect_no_longer_leans_to_dense_constellations():
    """QPSK at 4 dB: fitted per candidate it was 64QAM 19 times in 20; each scored at its own M2M4 Es/N0 it is QPSK"""
    got = [_detect("QPSK", 4, seed).mod for seed in range(20)]
    assert got.count("QPSK") >= 16

def test_detect_with_a_known_esn0():
    det = _detect("8PSK", 6, 3, esn0=6.0)
    assert det.mod == "8PSK" and det.how == "given" and det.given == 6.0
    assert _detect("8PSK", 6, 3).how == "moments" and _detect("8PSK", 15, 3).how == "fit"

def test_detect_says_it_doesnt_know():
    """min_evidence: noise alone is not named (no constellation beats Gaussian noise, or too little evidence)"""
    named = 0
    for seed in range(20):
        n = awgn(np.random.default_rng(seed), 8192, 1.0).astype(np.complex64)
        det = rp.sync.detect(n, 1e6, names=NAMES, baud=0.25e6, min_evidence=10.0)
        named += det.mod is not None
        assert det.best in NAMES
    assert named <= 1
    det = _detect("QPSK", 15, 0, min_evidence=10.0)
    assert det.mod == "QPSK" and det.evidence > 10
    with pytest.raises(rp.errors.NoLock):
        rp.sync.Lock.find(awgn(np.random.default_rng(1), 8192, 1.0).astype(np.complex64), 1e6, baud=0.25e6,
                          min_evidence=10.0)
