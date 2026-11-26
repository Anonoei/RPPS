"""How samples were received (pysdrlib's RxContext) on the stream: an inverting chain undone at the source, kept through
recordings, and reported by the signal profile"""
import dataclasses

import numpy as np
import pytest

import rpps as rp

pysdrlib = pytest.importorskip("pysdrlib")
if not hasattr(pysdrlib, "RxContext"):
    pytest.skip("pysdrlib before 0.5.1 has no RxContext", allow_module_level=True)
from pysdrlib.base import caps

LNB = caps.Chain(lo=5.15e9, invert=True, name="C-band LNB")
CTX = pysdrlib.RxContext(freq=3.9e9, rate=1e6, chain=LNB)

def tone(f, n=4096, Fs=1e6):
    return np.exp(2j*np.pi*f/Fs*np.arange(n)).astype(np.complex64)

def peak(x, Fs=1e6):
    return np.fft.fftfreq(len(x), 1/Fs)[np.argmax(np.abs(np.fft.fft(x)))]

def test_an_inverting_chain_is_undone_at_the_source():
    """The LO above the signal: a tone the radio gives at +100 kHz is 100 kHz under the centre over the air"""
    src = rp.source.Memory(tone(100e3), Fs=1e6, cf=3.9e9, context=CTX)
    blk = next(iter(src(4096)))
    assert src.conjugated and blk.stream.rx is CTX and blk.stream.rx_conjugated
    assert abs(peak(blk.data) + 100e3) < 500 and abs(peak(src.peek(4096)) + 100e3) < 500
    raw = rp.source.Memory(tone(100e3), Fs=1e6, cf=3.9e9, context=CTX, correct=False)
    assert not raw.conjugated and abs(peak(next(iter(raw(4096))).data) - 100e3) < 500
    blk.stream.restart()
    assert blk.stream.rx is CTX # (a restart keeps how the samples were received)

def test_a_chain_below_the_signal_changes_nothing():
    ctx = dataclasses.replace(CTX, chain=caps.Chain(lo=10.75e9))
    src = rp.source.Memory(tone(100e3), Fs=1e6, cf=11.7e9, context=ctx)
    assert not src.conjugated and abs(peak(next(iter(src(4096))).data) - 100e3) < 500

def test_a_recording_of_corrected_samples_says_so(tmp_path):
    src = rp.source.Memory(tone(100e3), Fs=1e6, cf=3.9e9, context=CTX)
    blk = next(iter(src(4096)))
    with rp.source.SigMFSink(tmp_path / "rec", Fs=1e6, cf=3.9e9) as sink:
        sink.write(blk)
    back = rp.source.SigMF(tmp_path / "rec")
    assert back.context.undone and back.context.chain == LNB and not back.conjugated # (not conjugated twice)
    assert np.allclose(next(iter(back(4096))).data, blk.data) and back.stream.rx_conjugated
    mem = rp.source.Memory.capture(rp.source.Memory(tone(100e3), Fs=1e6, cf=3.9e9, context=CTX), 4096)
    assert mem.context.undone and not mem.conjugated and abs(peak(mem.samples) + 100e3) < 500

def test_a_pysdrlib_recording_is_read_the_right_way_round(tmp_path):
    with pysdrlib.FileSink(tmp_path / "raw.sigmf", "cf32", context=CTX) as s:
        s.write(tone(100e3))
    src = rp.source.SigMF(tmp_path / "raw.sigmf")
    assert src.context.inverted and src.cf == 3.9e9 and abs(peak(next(iter(src(4096))).data) + 100e3) < 500

def test_a_device_gives_its_context():
    sdr = pysdrlib.device("sim", realtime=False)
    sdr.open()
    try:
        sdr.set_chains([LNB])
        src = rp.source.Device(sdr, cf=3.9e9, Fs=1e6)
        blk = next(iter(src(4096)))
        assert blk.stream.rx.chain == LNB and src.conjugated
    finally:
        sdr.close()

@pytest.mark.kernels
def test_the_profile_places_the_carrier_over_the_air(tmp_path):
    """A DVB-S carrier 1.7 MHz above a 3.9 GHz centre, received through the inverting LNB: the profile puts it at the
    frequency over the air, says the reference is the radio's own, and that our chain's inversion was undone"""
    rng = np.random.default_rng(0)
    ts = np.zeros((900, 188), np.uint8)
    ts[:, 0] = 0x47
    ts[:, 1:] = rng.integers(0, 256, (900, 187))
    w = rp.profile.load("dvbs/qpsk_3_4", sps=8).tx()(rp.Data(np.unpackbits(ts.reshape(-1)).astype(bool), Fs=1e6, sps=1, flush=True))
    x = np.asarray(w.data)*np.exp(2j*np.pi*1.7e6/8e6*np.arange(len(w.data)))
    x = (x + 0.05*(rng.standard_normal(len(x)) + 1j*rng.standard_normal(len(x)))).astype(np.complex64)
    ctx = dataclasses.replace(CTX, rate=8e6)
    src = rp.source.Memory(np.conj(x), Fs=8e6, cf=3.9e9, context=ctx) # (as the radio gives it: mirrored)
    p = rp.analysis.profile(src, freq=1.7e6, blind=False)[0]
    f = next(s for s in p.steps if s.kind == "frequency")
    assert abs(f.detail["freq"] - 3.9017e9) < 2e3 and f.status == "guess" and f.detail["ppm_accuracy"] is None
    assert "the reference's accuracy and the LO's accuracy not known" in f.note
    spec = next(s for s in p.steps if s.kind == "spectrum")
    assert spec.what == "normal" and "conjugated at the source" in spec.note
    assert p.solved and p.profile_name == "dvbs/qpsk_3_4"

def test_what_a_frequency_is_worth_comes_from_the_context():
    """The reference's accuracy at the carrier's IF, and the LO's, added in quadrature; unknown parts named"""
    from rpps.analysis.signal_profile._receive import accuracy
    direct = pysdrlib.RxContext(freq=751e6, ppm=-14.0, ppm_accuracy=0.1)
    sf, ppm_acc, unknown = accuracy(direct, 751e6)
    assert sf == pytest.approx(75.1) and ppm_acc == 0.1 and unknown == []
    lnb = pysdrlib.RxContext(freq=11.7e9, ppm_accuracy=0.5, chain=caps.Chain(lo=9.75e9, lo_accuracy=2e3))
    sf, _, _ = accuracy(lnb, 11.7e9) # the IF is 1.95 GHz: 975 Hz from the reference, 2 kHz from the LO
    assert sf == pytest.approx((975.0**2 + 2e3**2)**0.5)
    assert accuracy(dataclasses.replace(lnb, chain=caps.Chain(lo=9.75e9)), 11.7e9)[2] == ["the LO's accuracy"]
    assert accuracy(None, 1e9) == (None, None, ["the reference's accuracy"])
