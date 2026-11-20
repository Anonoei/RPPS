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
    return rp.signal.filters.designer.designer(101, 7.5e3, fs=250e3)

STAGES = {
    "FMDemod": (lambda: rp.wave.FMDemod(), dict(Fs=250e3)),
    "ShiftFreq": (lambda: rp.signal.filters.ShiftFreq(-57e3), dict(Fs=250e3)),
    "ShiftFreqIdx": (lambda: rp.signal.filters.ShiftFreqIdx(0.01), dict(Fs=250e3)),
    "FIR": (lambda: rp.signal.filters.filters.ConvFilter(taps()), dict(Fs=250e3)),
    "ShiftTime": (lambda: rp.signal.filters.ShiftTime(0.3), dict(Fs=250e3)),
    "Decimate": (lambda: rp.signal.sample.down.Decimate(10), dict(Fs=250e3, sps=10)),
    "FractionalSpline": (lambda: rp.signal.sample.down.FractionalSpline(0.37), dict(Fs=250e3, sps=10)),
    "Polyphase": (lambda: rp.signal.sample.re.Polyphase(19, 25), dict(Fs=25e3, sps=25)),
    "FractionalPolyphase": (lambda: rp.signal.sample.FractionalPolyphase(0.5381, cutoff=3.7e6, shift=-0.5e6), dict(Fs=20e6, sps=2, cf=604e6)),
    "SincUp": (lambda: rp.signal.sample.up.Sinc(4), dict(Fs=25e3, sps=1)),
    "Biphase": (lambda: rp.signal.filters.ShapingBiphase(), dict(Fs=19e3, sps=16)),
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
    stage = rp.signal.filters.ShiftFreq(1e3)
    a = rp.Data(x[:100], Fs=1e5) + stage
    b = rp.Data(x[:100], Fs=1e5) + stage # a new stream: the oscillator starts again
    assert np.array_equal(a.data, b.data)

def test_rates_follow_the_stages(x):
    d = rp.Data(x[:5000], Fs=250e3, sps=10, cf=100e6)
    out = d + rp.signal.filters.ShiftFreq(25e3) + rp.signal.sample.down.Decimate(5)
    assert out.Fs == 50e3 and out.sps == 2 and out.cf == 100e6 + 25e3
    assert d.Fs == 250e3 and d.cf == 100e6 # The input is untouched

def test_group_delay_is_per_block_not_summed(x):
    lpf = rp.signal.filters.low_pass(101, 5e3, 250e3)
    stream, delays = None, []
    for i in range(3):
        d = rp.Data(x[i*1000:(i+1)*1000], Fs=250e3, stream=stream)
        stream = d.stream
        delays.append((d + lpf).delay)
    assert delays[0] == pytest.approx(50/250e3) and len(set(delays)) == 1

def test_pipelines_compose_and_match_the_operators(x):
    a, b, c = rp.signal.filters.ShiftFreq(1e3), rp.signal.filters.low_pass(51, 40e3, 250e3), rp.signal.sample.down.Decimate(4)
    rx = a + b + c
    assert isinstance(rx, rp.base.Pipeline) and len(rx) == 3
    ref = rp.Data(x[:4000], Fs=250e3) + a + b + c
    out = rp.Data(x[:4000], Fs=250e3) + rx
    assert np.array_equal(ref.data, out.data) and out.Fs == 62.5e3
    assert len(a + (b + c)) == 3 and len((a + b) + c) == 3

def test_digital_pipelines_encode_forward_and_decode_backward():
    m = rp.symbol.load("QPSK"); m.set_mapping(m.maps[0])
    ecc = rp.code.load("block/hamming/7_4")
    scr = rp.code.scram.load("v35")
    link = scr * ecc * m
    msg = rp.Data(b"Hello World!")
    syms = msg * link
    assert np.array_equal(syms.data, (msg * scr * ecc * m).data)
    assert (syms / link).as_bytes().hex == b"Hello World!".hex()

def test_stages_reject_digital_data():
    with pytest.raises(rp.errors.StreamError):
        rp.Data(b"abc") + rp.signal.filters.ShiftFreq(1.0)

def test_blocks_flow_through_pipelines(tmp_path, x):
    path = tmp_path / "a.iq"
    x[:5000].tofile(path)
    rx = rp.signal.filters.ShiftFreq(1e3) + rp.signal.sample.down.Decimate(5)
    src = rp.source.File("cf64", path, Fs=100e3, cf=1e6)
    got = (src(777) + rx).collect() # 777 doesn't divide 5000: a short last block
    ref = (rp.Data(x[:5000], Fs=100e3) + rx).data
    assert len(got) == len(ref) and np.allclose(got, ref, atol=1e-5)
    assert src.percent == pytest.approx(1.0)

def test_sink_round_trip(tmp_path, x):
    src = tmp_path / "a.iq"
    (x[:3000]*0.2).astype(np.complex64).tofile(src)
    for fmt, tol in (("ci32t", 1/32768), ("ci16t", 1/128), ("cf64", 1e-7)):
        out = tmp_path / f"b.{fmt}"
        with rp.source.Sink(fmt, out) as sink:
            rp.source.File("cf64", src)(1000) >> sink
        back = rp.source.File(fmt, out).read(3000)
        assert np.abs(back - x[:3000]*0.2).max() <= tol

def test_file_seek_resumes_where_it_was(tmp_path, x):
    path = tmp_path / "a.iq"
    x[:1000].tofile(path)
    f = rp.source.File("cf64", path)
    f.read(100)
    f.seek(400)
    assert np.array_equal(f.read(10), x[400:410])

CHANNEL = {
    "AWGN": (lambda: rp.signal.channel.AWGN(10, seed=1), dict(Fs=1e4)),
    "CFO": (lambda: rp.signal.channel.CFO(300), dict(Fs=1e4)),
    "Fading": (lambda: rp.signal.channel.Fading(50, seed=2), dict(Fs=1e4)),
    "ClockDrift": (lambda: rp.signal.channel.ClockDrift(200), dict(Fs=1e4)),
    "TimingOffset": (lambda: rp.signal.channel.TimingOffset(0.4), dict(Fs=1e4)),
    "Multipath": (lambda: rp.signal.channel.Multipath([1, 0.4j], [0, 3]), dict(Fs=1e4)),
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
    "FractionalSpline cubic": (lambda: rp.signal.sample.down.FractionalSpline(0.73, "cubic"), dict(Fs=1e4, sps=8)),
    "CMA": (lambda: rp.sync.equalize.CMA(7, 0.003, sps=2), dict(Fs=1e4, sps=2)),
    "LMS": (lambda: rp.sync.equalize.LMS(5, 0.01, sps=1, points=[1, -1, 1j, -1j]), dict(Fs=1e4, sps=1)),
    "AMDemod": (lambda: rp.wave.AMDemod(), dict(Fs=1e4)),
    "SSBDemod": (lambda: rp.wave.SSBDemod(2000, "usb", 51), dict(Fs=1e4)),
    "Squelch": (lambda: rp.sync.Squelch(-3.0), dict(Fs=1e4)),
    "DCBlock": (lambda: rp.signal.DCBlock(), dict(Fs=1e4)),
    "Deemphasis": (lambda: rp.signal.Deemphasis(), dict(Fs=1e4)),
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

@pytest.mark.kernels
def test_acquire_fec_does_not_depend_on_block_size():
    """Coded symbols (K=7 rate 1/2 on QPSK, a quarter turn off) with the modulation already detected"""
    from rpps.sync import Detection
    rng = np.random.default_rng(5)
    cw = rp.code.load("conv/k7_133_171/1_2").encoder.encode(rng.integers(0, 2, 12000).astype(bool))
    s = rp.symbol.load("qpsk").modulate(cw)*1j
    x = (s + 0.1*(rng.standard_normal(len(s)) + 1j*rng.standard_normal(len(s)))).astype(np.complex64)
    stage = rp.sync.AcquireFEC(window=4096, skip=256, max_wait=8192)
    outs, names = [], set()
    for sp in cuts(len(x)):
        stream, out, i = None, [], 0
        for b in sp:
            d = rp.Data(x[i:i+b], Fs=1e4, sps=1, stream=stream)
            stream = d.stream
            if stream.det is None:
                stream.det = Detection(mod="qpsk", baud=1e4, sps=1.0)
            out.append((d + stage).data)
            i += b
        # The code, or the same code with its generators swapped (I and Q swapped): the one a shipped profile uses wins
        assert stream.est["fec"].name in ("k7_133_171/1_2", "k7_171_133/1_2")
        names.add(stream.est["fec"].name)
        outs.append(np.concatenate(out))
    assert len(names) == 1
    n = min(len(o) for o in outs)
    assert n > 8000 and all(np.array_equal(o[:n], outs[0][:n]) for o in outs)

@pytest.mark.parametrize("native", [True, False])
@pytest.mark.parametrize("ratio,taps", [(4, 65), (8, 129), (3, 10)])
def test_interpolate_is_zeros_then_filter(ratio, taps, native):
    """The polyphase interpolator gives Zeros + the filter, however the stream is cut"""
    from rpps import _accel
    saved = _accel.native
    _accel.native = saved if native else None
    try:
        rng = np.random.default_rng(ratio)
        syms = (rng.standard_normal(700) + 1j*rng.standard_normal(700)).astype(np.complex64)
        pulse = rp.signal.filters.ShapingRRC(taps, ratio)
        def run(stages, cut):
            out, st, i = [], None, 0
            for b in cut:
                d = rp.Data(syms[i:i+b], stream=st, Fs=1e3, sps=1)
                st = d.stream
                for s in stages:
                    d = d + s
                out.append(d.data)
                i += b
            return np.concatenate(out), d
        ref, dr = run([rp.signal.sample.up.Zeros(ratio), pulse], [700])
        for cut in ([700], [1, 0, 300, 399]):
            y, d = run([rp.signal.sample.up.Interpolate(ratio, pulse)], cut)
            np.testing.assert_allclose(y, ref, atol=1e-5)
            assert d.Fs == dr.Fs and d.sps == dr.sps and d.delay == pytest.approx(dr.delay)
    finally:
        _accel.native = saved

def _tee_run(tee, x, cut, **rates):
    outs, st, i = {}, None, 0
    for b in cut:
        d = rp.Data(x[i:i+b], stream=st, **rates)
        st = d.stream
        for k, v in (d + tee).items():
            outs.setdefault(k, []).append(np.asarray(v.data))
        i += b
    return {k: np.concatenate(v) for k, v in outs.items()}, v

def test_tee_branches_share_stages_not_state():
    """One stage in two branches has a state in each: each branch gives what it gives alone, however the stream is cut"""
    rng = np.random.default_rng(9)
    x = (rng.standard_normal(3000) + 1j*rng.standard_normal(3000)).astype(np.complex64)
    lp = rp.signal.filters.low_pass(63, 0.2)
    shift = rp.signal.filters.ShiftFreq(0.05)
    tee = rp.Tee(plain=lp, shifted=shift + lp)
    alone_plain = np.asarray((rp.Data(x, Fs=1.0) + rp.signal.filters.low_pass(63, 0.2)).data)
    alone_shift = np.asarray((rp.Data(x, Fs=1.0) + rp.signal.filters.ShiftFreq(0.05) + rp.signal.filters.low_pass(63, 0.2)).data)
    for cut in ([3000], [1, 0, 999, 2000]):
        out, last = _tee_run(tee, x, cut, Fs=1.0)
        np.testing.assert_allclose(out["plain"], alone_plain, atol=1e-5)
        np.testing.assert_allclose(out["shifted"], alone_shift, atol=1e-5)
    assert last.state(shift) is not None and last.cf == pytest.approx(0.05)

def test_tee_needs_branches_and_data():
    with pytest.raises(ValueError):
        rp.Tee()
    assert rp.Tee(a=rp.signal.DCBlock()).__radd__(np.zeros(3)) is NotImplemented

def test_wav_out(tmp_path):
    import wave
    t = np.arange(4800)/48000
    x = 0.5*np.sin(2*np.pi*440*t)
    n = rp.source.write_wav(tmp_path/"a.wav", x, 48000, peak=0.8)
    with wave.open(str(tmp_path/"a.wav")) as w:
        y = np.frombuffer(w.readframes(n), np.int16)
        assert w.getframerate() == 48000 and w.getnchannels() == 1
    assert abs(np.abs(y).max() - 0.8*32767) <= 1
    with rp.source.WavSink(tmp_path/"b.wav") as s:
        st = None
        for i in range(0, len(x), 1000):
            d = rp.Data(x[i:i+1000].astype(np.float32), stream=st, Fs=48000)
            st = d.stream
            d >> s
    with wave.open(str(tmp_path/"b.wav")) as w:
        z = np.frombuffer(w.readframes(len(x)), np.int16)
    assert np.array_equal(z, np.round(x*32767).astype(np.int16))

def test_source_clock_error_and_what_it_gives():
    src = rp.source.Memory(np.zeros(100, np.complex64), Fs=1e6, cf=100e6)
    src.ppm = 15.0
    b = next(iter(src(50)))
    assert b.Fs == pytest.approx(1e6*(1 + 15e-6)) and b.cf == pytest.approx(100e6*(1 + 15e-6))
    assert src.REP == rp.Rep.SAMPLES and not src.APPLIED
    class Timed(rp.source.Memory):
        REP, APPLIED = rp.Rep.SYMBOLS, frozenset({"timing"})
    p = rp.profile.load("ieee802154/2450")
    names = lambda r: [type(op).__name__ for _, op in r.ops]
    assert names(p.rx(source=Timed(np.zeros(4, np.complex64)))) == names(p.rx(start=rp.Rep.SYMBOLS, applied={"timing"}))

def test_interpolate_sends_the_last_pulse_whole():
    """A flushed transmission ends with its filter's tail: the last symbol is as clean as any other"""
    pulse = rp.signal.filters.ShapingRRC(65, 4)
    syms = np.exp(1j*np.pi/4*(2*np.random.default_rng(3).integers(0, 4, 200) + 1)).astype(np.complex64)
    tx = np.asarray((rp.Data(syms, Fs=1e3, sps=1, flush=True) + rp.signal.sample.up.Interpolate(4, pulse)).data)
    assert len(tx) == 4*200 + 4*16
    h = pulse.pulse
    rx = np.convolve(tx, h)/np.sum(h*h)
    got = rx[len(h) - 1::4][:200]
    assert np.max(np.abs(got - syms)) < 0.05 # the last symbols too
