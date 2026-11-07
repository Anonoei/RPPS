"""Stages give the same stream however it is cut into blocks, and the operators drive them"""
import numpy as np
import pytest

import rpps as rp

def run(make, x, blocks, **rates):
    """x cut into `blocks`, each as a block of one stream, through a fresh stage"""
    stage = make()
    stream, outs, i = None, [], 0
    for b in blocks:
        d = rp.Data(x[i:i+b], stream=stream, **rates)
        stream = d.stream
        outs.append((d + stage).data)
        i += b
    return np.concatenate(outs)

def cuts(n):
    return ([n], [1000]*(n//1000), [777]*(n//777) + [n % 777], [1]*50 + [n-50])

@pytest.fixture(scope="module")
def x():
    rng = np.random.default_rng(0)
    return (rng.standard_normal(20000) + 1j*rng.standard_normal(20000)).astype(np.complex64)

def taps():
    return rp.filters.designer.designer(101, 7.5e3, fs=250e3)

STAGES = {
    "FMDemod": (lambda: rp.mod.FMDemod(), dict(Fs=250e3)),
    "ShiftFreq": (lambda: rp.filters.ShiftFreq(-57e3), dict(Fs=250e3)),
    "ShiftFreqIdx": (lambda: rp.filters.ShiftFreqIdx(0.01), dict(Fs=250e3)),
    "FIR": (lambda: rp.filters.filters.ConvFilter(taps()), dict(Fs=250e3)),
    "ShiftTime": (lambda: rp.filters.ShiftTime(0.3), dict(Fs=250e3)),
    "Decimate": (lambda: rp.sample.down.Decimate(10), dict(Fs=250e3, sps=10)),
    "FractionalSpline": (lambda: rp.sample.down.FractionalSpline(0.37), dict(Fs=250e3, sps=10)),
    "Polyphase": (lambda: rp.sample.re.Polyphase(19, 25), dict(Fs=25e3, sps=25)),
    "SincUp": (lambda: rp.sample.up.Sinc(4), dict(Fs=25e3, sps=1)),
    "Biphase": (lambda: rp.filters.ShapingBiphase(), dict(Fs=19e3, sps=16)),
    "AGC": (lambda: rp.sync.AGC(1.0, 0.01), dict(Fs=1e3)),
    "Costas": (lambda: rp.sync.Costas(0.05, 0.001, 4), dict(Fs=1e3, sps=1)),
}

@pytest.mark.parametrize("name", STAGES)
def test_output_does_not_depend_on_block_size(name, x):
    make, rates = STAGES[name]
    x = x[:4000] if name == "Costas" else x # A per-sample Python loop
    ref = run(make, x, [len(x)], **rates)
    for sp in cuts(len(x))[1:]:
        y = run(make, x, sp, **rates)
        assert len(y) == len(ref), f"{len(y)} != {len(ref)} samples for blocks {sp[:2]}.."
        assert np.allclose(y, ref, atol=1e-5)

def test_mm_does_not_depend_on_block_size():
    rng = np.random.default_rng(3)
    sym = rng.choice([-1, 1], 3000) + 1j*rng.choice([-1, 1], 3000)
    sig = np.repeat(sym, 8).astype(np.complex64)
    sig = np.convolve(sig, np.ones(8)/8, mode="same").astype(np.complex64)
    ref = run(lambda: rp.sync.time.MM(), sig, [len(sig)], Fs=8e3, sps=8)
    y = run(lambda: rp.sync.time.MM(), sig, [4000, 4000, len(sig)-8000], Fs=8e3, sps=8)
    n = min(len(ref), len(y))
    assert n > 2000 and abs(len(ref) - len(y)) <= 2
    assert np.allclose(ref[:n], y[:n], atol=1e-4)

def test_a_stage_is_reusable_across_streams(x):
    stage = rp.filters.ShiftFreq(1e3)
    a = rp.Data(x[:100], Fs=1e5) + stage
    b = rp.Data(x[:100], Fs=1e5) + stage # a new stream: the oscillator starts again
    assert np.array_equal(a.data, b.data)

def test_rates_follow_the_stages(x):
    d = rp.Data(x[:5000], Fs=250e3, sps=10, cf=100e6)
    out = d + rp.filters.ShiftFreq(25e3) + rp.sample.down.Decimate(5)
    assert out.Fs == 50e3 and out.sps == 2 and out.cf == 100e6 + 25e3
    assert d.Fs == 250e3 and d.cf == 100e6 # The input is untouched

def test_group_delay_is_per_block_not_summed(x):
    lpf = rp.filters.low_pass(101, 5e3, 250e3)
    stream, delays = None, []
    for i in range(3):
        d = rp.Data(x[i*1000:(i+1)*1000], Fs=250e3, stream=stream)
        stream = d.stream
        delays.append((d + lpf).delay)
    assert delays[0] == pytest.approx(50/250e3) and len(set(delays)) == 1

def test_pipelines_compose_and_match_the_operators(x):
    a, b, c = rp.filters.ShiftFreq(1e3), rp.filters.low_pass(51, 40e3, 250e3), rp.sample.down.Decimate(4)
    rx = a + b + c
    assert isinstance(rx, rp.base.Pipeline) and len(rx) == 3
    ref = rp.Data(x[:4000], Fs=250e3) + a + b + c
    out = rp.Data(x[:4000], Fs=250e3) + rx
    assert np.array_equal(ref.data, out.data) and out.Fs == 62.5e3
    assert len(a + (b + c)) == 3 and len((a + b) + c) == 3

def test_digital_pipelines_encode_forward_and_decode_backward():
    m = rp.mod.load("QPSK"); m.set_mapping(m.get_maps()[0])
    ecc = rp.coding.load("block", "hamming/7_4")
    scr = rp.scram.load("fdt", "v35")
    link = scr * ecc * m
    msg = rp.Data(b"Hello World!")
    syms = msg * link
    assert np.array_equal(syms.data, (msg * scr * ecc * m).data)
    assert (syms / link).as_bytes().hex == b"Hello World!".hex()

def test_stages_reject_digital_data():
    with pytest.raises(rp.errors.StreamError):
        rp.Data(b"abc") + rp.filters.ShiftFreq(1.0)

def test_blocks_flow_through_pipelines(tmp_path, x):
    path = tmp_path / "a.iq"
    x[:5000].tofile(path)
    rx = rp.filters.ShiftFreq(1e3) + rp.sample.down.Decimate(5)
    src = rp.serial.File("cf64", path, Fs=100e3, cf=1e6)
    got = (src(777) + rx).collect() # 777 doesn't divide 5000: a short last block
    ref = (rp.Data(x[:5000], Fs=100e3) + rx).data
    assert len(got) == len(ref) and np.allclose(got, ref, atol=1e-5)
    assert src.percent == pytest.approx(1.0)

def test_sink_round_trip(tmp_path, x):
    src = tmp_path / "a.iq"
    (x[:3000]*0.2).astype(np.complex64).tofile(src)
    for fmt, tol in (("ci32t", 1/32768), ("ci16t", 1/128), ("cf64", 1e-7)):
        out = tmp_path / f"b.{fmt}"
        with rp.serial.Sink(fmt, out) as sink:
            rp.serial.File("cf64", src)(1000) >> sink
        back = rp.serial.File(fmt, out).read(3000)
        assert np.abs(back - x[:3000]*0.2).max() <= tol

def test_file_seek_resumes_where_it_was(tmp_path, x):
    path = tmp_path / "a.iq"
    x[:1000].tofile(path)
    f = rp.serial.File("cf64", path)
    f.read(100)
    f.seek(400)
    assert np.array_equal(f.read(10), x[400:410])

CHANNEL = {
    "AWGN": (lambda: rp.channel.AWGN(10, seed=1), dict(Fs=1e4)),
    "CFO": (lambda: rp.channel.CFO(300), dict(Fs=1e4)),
    "Fading": (lambda: rp.channel.Fading(50, seed=2), dict(Fs=1e4)),
    "ClockDrift": (lambda: rp.channel.ClockDrift(200), dict(Fs=1e4)),
    "TimingOffset": (lambda: rp.channel.TimingOffset(0.4), dict(Fs=1e4)),
    "Multipath": (lambda: rp.channel.Multipath([1, 0.4j], [0, 3]), dict(Fs=1e4)),
}

@pytest.mark.parametrize("name", CHANNEL)
def test_channel_stages_do_not_depend_on_block_size(name, x):
    make, rates = CHANNEL[name]
    x = x[:6000]
    ref = run(make, x, [len(x)], **rates)
    for sp in cuts(len(x))[1:]:
        y = run(make, x, sp, **rates)
        assert len(y) == len(ref)
        assert np.allclose(y, ref, atol=1e-4)

MORE = {
    "FractionalSpline cubic": (lambda: rp.sample.down.FractionalSpline(0.73, "cubic"), dict(Fs=1e4, sps=8)),
    "CMA": (lambda: rp.equalize.CMA(7, 0.003, sps=2), dict(Fs=1e4, sps=2)),
    "LMS": (lambda: rp.equalize.LMS(5, 0.01, sps=1, points=[1, -1, 1j, -1j]), dict(Fs=1e4, sps=1)),
    "AMDemod": (lambda: rp.demod.AMDemod(), dict(Fs=1e4)),
    "SSBDemod": (lambda: rp.demod.SSBDemod(2000, "usb", 51), dict(Fs=1e4)),
    "Squelch": (lambda: rp.demod.Squelch(-3.0), dict(Fs=1e4)),
    "DCBlock": (lambda: rp.demod.DCBlock(), dict(Fs=1e4)),
    "Deemphasis": (lambda: rp.demod.Deemphasis(), dict(Fs=1e4)),
}

@pytest.mark.parametrize("name", MORE)
def test_more_stages_do_not_depend_on_block_size(name, x):
    make, rates = MORE[name]
    x = x[:6000]
    ref = run(make, x, [len(x)], **rates)
    for sp in cuts(len(x))[1:]:
        y = run(make, x, sp, **rates)
        assert len(y) == len(ref), f"{len(y)} != {len(ref)}"
        assert np.allclose(y, ref, atol=1e-4)
