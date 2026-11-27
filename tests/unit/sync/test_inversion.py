"""Spectral inversion: rp.sync.Inversion (known symbols that turn backwards) and the code's own say (CodeDetection.inverted)"""
import numpy as np
import pytest

import rpps as rp
from rpps.analysis import identify as F
from rpps.signal import channel as ch
from rpps.sync.inversion import sense, handed

SOF = rp.symbol.load_header("dvbs2_pl").preamble

def framed(rng, n=40, gap=3000, offset=0.003, noise=0.2):
    """QPSK data with the DVB-S2 SOF every `gap` symbols, turned by an offset and a phase, in noise"""
    data = np.exp(1j*np.pi/4*(2*rng.integers(0, 4, n*gap) + 1))
    z = np.concatenate([np.concatenate((SOF, data[k*gap:(k + 1)*gap])) for k in range(n)])
    z = z*np.exp(2j*np.pi*offset*np.arange(len(z)) + 0.7j)
    return (z + noise*(rng.standard_normal(len(z)) + 1j*rng.standard_normal(len(z)))).astype(np.complex64)

def test_the_preamble_turning_backwards_means_an_inverted_spectrum():
    rng = np.random.default_rng(3)
    data = np.exp(1j*np.pi/4*(2*rng.integers(0, 4, 3000) + 1)) # QPSK: the same conjugated, says nothing on its own
    z = np.concatenate([data[:1000], SOF, data[1000:2000], SOF, data[2000:]])
    z = z*np.exp(2j*np.pi*0.003*np.arange(len(z)) + 0.7j) + 0.2*(rng.standard_normal(len(z)) + 1j*rng.standard_normal(len(z)))
    assert sense(z, SOF) is False and sense(np.conj(z), SOF) is True
    assert sense(data, SOF) is None

def test_a_sequence_that_never_turns_cannot_tell():
    bpsk = np.array([1, -1, -1, 1, -1, 1, 1, 1], complex)
    assert handed(SOF) and not handed(bpsk) and not handed(bpsk*np.exp(0.3j)) # (a constant phase is no turn)
    assert sense(np.tile(bpsk, 20), bpsk) is None
    with pytest.raises(ValueError, match="turn"):
        rp.sync.Inversion(bpsk)

def run(stage, x, cuts):
    out, st, i = [], None, 0
    for k, n in enumerate(cuts):
        d = rp.Data(x[i:i + n], sps=1, Fs=1e6, stream=st, flush=k == len(cuts) - 1)
        st, i = d.stream, i + n
        out.append(np.asarray((d + stage).data))
    return np.concatenate(out), st

@pytest.mark.parametrize("inverted", [False, True])
def test_inversion_conjugates_back_however_the_stream_is_cut(inverted):
    z = framed(np.random.default_rng(1))
    x = np.conj(z) if inverted else z
    whole, st = run(rp.sync.Inversion(SOF), x, [len(x)])
    assert np.array_equal(whole, z) # Every symbol, the held ones too, the right way round
    assert st.est["inversion"]["inverted"] is inverted and st.est["inversion"]["matches"] >= 30
    assert [f for _, f, *_ in st.control.published] == [{"inverted": inverted}] # (a profile logs them: control.fields())
    rng = np.random.default_rng(2)
    cuts = list(rng.integers(0, 9000, 30)) + [0, 1]
    cuts.append(len(x) - sum(cuts))
    assert np.array_equal(run(rp.sync.Inversion(SOF), x, cuts)[0], whole)

def test_inversion_holds_then_lets_go_without_a_preamble():
    rng = np.random.default_rng(4)
    x = np.exp(1j*np.pi/4*(2*rng.integers(0, 4, 50000) + 1)).astype(np.complex64)
    out, st = run(rp.sync.Inversion(SOF, hold=8192), x, [20000, 30000])
    assert np.array_equal(out, x) and st.est["inversion"]["inverted"] is None
    levels = rng.standard_normal(1000).astype(np.float32) # Real levels: inversion is their polarity, a frame sync's
    assert np.array_equal(run(rp.sync.Inversion(SOF), levels, [1000])[0], levels)

def test_the_sense_changes_only_on_enough_evidence():
    rng = np.random.default_rng(5)
    z = framed(rng, n=20)
    x = np.concatenate((z, np.conj(framed(rng, n=20)))) # The second half conjugated (a mixer switched)
    out, st = run(rp.sync.Inversion(SOF), x, [len(x)])
    assert [f["inverted"] for _, f, *_ in st.control.published] == [False, True]
    half = len(z)
    assert np.array_equal(out[:half], x[:half]) # Before the switch as received
    assert np.allclose(out[-half//2:], np.conj(x[-half//2:])) # Well after it, conjugated back

def test_dvbs2_decodes_a_conjugated_capture():
    p = rp.profile.load("dvbs2/normal")
    frames = [{"modcod": m, "pilots": 1} for m in (4, 13, 19)]
    t = rp.profile.table("dvbs2/modcod")
    data = np.random.default_rng(0).integers(0, 2, sum(t.row({**f, "short": 0})["kbch"] - 80 for f in frames)).astype(bool)
    o = p.tx()(rp.Data(data, Fs=4e6, sps=1, flush=True), fields=frames)
    x = rp.Data(np.concatenate((np.zeros(4000, np.complex64), o.data, np.zeros(4000, np.complex64))), Fs=o.Fs, sps=4, flush=True)
    x = x + ch.CFO(2000) + ch.PhaseOffset(1.1) + ch.AWGN(esn0_db=20, power=float(np.mean(np.abs(o.data)**2)), seed=3)
    d = rp.Data(np.conj(np.asarray(x.data)), Fs=o.Fs, sps=4, flush=True)
    got = np.asarray(p.rx()(d).data).astype(bool)
    assert np.array_equal(got, data) and d.stream.est["inversion"]["inverted"] is True

@pytest.fixture(scope="module")
def dvbs():
    """DVB-S symbols (dvbs/qpsk_3_4 transmit side, no pulse shaping)"""
    rng = np.random.default_rng(1)
    p = rp.profile.load("dvbs/qpsk_3_4")
    ts = np.zeros((220, 188), np.uint8)
    ts[:, 0] = 0x47
    ts[:, 1:] = rng.integers(0, 256, (220, 187))
    d = rp.Data(np.unpackbits(ts.reshape(-1)).astype(bool), flush=True)
    for how, op in p.tx():
        if how == "*":
            d = d * op
    return np.asarray(d.data).astype(np.complex64)

@pytest.mark.kernels
@pytest.mark.parametrize("inverted", [False, True])
def test_the_code_says_the_sense_by_a_standards_map(dvbs, inverted):
    """DVB-S's code decodes on its profile's map only one way round: the conjugate is the spectrum's, not a choice of map"""
    rng = np.random.default_rng(2)
    y = dvbs[5:]*np.exp(1j*np.pi/2) # Unknown alignment, the carrier a quarter turn off
    y = np.conj(y) if inverted else y
    f = F.identify((y + 0.12*(rng.standard_normal(len(y)) + 1j*rng.standard_normal(len(y))))[:2**17], "qpsk")
    assert f.profile_name == "dvbs/qpsk_3_4" and f.inverted is inverted
