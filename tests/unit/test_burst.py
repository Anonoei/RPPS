"""TDMA bursts demodulated on their shared preamble (rpps.analysis.spectrum.burst)"""
import numpy as np

import rpps as rp
from rpps.analysis.spectrum import burst
from rpps.analysis.framing import frames as FR

def tdma(rng, frames=3, slots=(0, 2, 3, 5), fs=2e6, baud=250e3, frame=4.8e-3, slot=0.6e-3, noise=0.08):
    q = rp.symbol.load("qpsk").points
    pre = q[rng.integers(0, 4, 26)]
    x = np.zeros(int((frames + 0.2)*frame*fs), complex)
    sent = []
    for f in range(frames):
        for s in slots:
            syms = np.concatenate((pre, q[rng.integers(0, 4, 104)]))
            w = FR.word_waveform(syms, fs/baud)
            w = w*np.exp(1j*rng.uniform(0, 2*np.pi) + 2j*np.pi*rng.uniform(-3e3, 3e3)/fs*np.arange(len(w)))
            a = int((f*frame + s*slot)*fs) + int(rng.integers(0, 6))
            x[a:a + len(w)] += w
            sent.append(syms)
    x += noise*(rng.standard_normal(len(x)) + 1j*rng.standard_normal(len(x)))
    return x, fs, baud, sent, q

def test_every_burst_on_its_own_carrier_and_phase():
    x, fs, baud, sent, q = tdma(np.random.default_rng(7))
    r = burst.demodulate(x, fs)
    assert r.mod == "qpsk" and abs(r.baud - baud) < 200 and len(r.preamble) >= 24
    assert abs(r.frame - 4.8e-3) < 2e-5 and len(r.slots) == 4
    assert [b.slot for b in r.bursts] == [0, 1, 2, 3]*3
    rot = set()
    for b, syms in zip(r.bursts, sent):
        assert b.ok
        dec = q[np.argmin(np.abs(b.symbols[:, None] - q[None, :]), axis=1)]
        # the found preamble starts a symbol in (its first step); one rotation for every burst
        n = min(len(dec), len(syms) - 1)
        errs = [np.mean(np.abs(dec[:n] - syms[1:n + 1]*1j**k) > 1e-6) for k in range(4)]
        assert min(errs) == 0
        rot.add(int(np.argmin(errs)))
    assert len(rot) == 1

def test_burst_kinds_each_on_its_own_preamble():
    """A network's burst types (here two lengths), each with its own unique word: what all the bursts share is a symbol or
    two, so each kind's preamble is found from its own bursts and every burst acquired on its kind's"""
    rng = np.random.default_rng(3)
    q = rp.symbol.load("qpsk").points
    fs, baud = 2e6, 250e3
    pres = [q[rng.integers(0, 4, 40)], q[rng.integers(0, 4, 40)]]
    x = np.zeros(int(0.03*fs), complex)
    sent, at = [], 2000
    for i in range(12):
        kind = i % 2
        syms = np.concatenate((pres[kind], q[rng.integers(0, 4, 200 if kind == 0 else 450)]))
        w = FR.word_waveform(syms, fs/baud)
        w = w*np.exp(1j*rng.uniform(0, 2*np.pi) + 2j*np.pi*rng.uniform(-2e3, 2e3)/fs*np.arange(len(w)))
        x[at:at + len(w)] += w
        sent.append((kind, syms))
        at += len(w) + 600
    x += 0.06*(rng.standard_normal(len(x)) + 1j*rng.standard_normal(len(x)))
    r = burst.demodulate(x, fs)
    assert len(r.preambles) == 2 and all(len(p) >= 32 for p in r.preambles)
    assert len(r.bursts) == 12 and all(b.ok for b in r.bursts)
    kinds = [b.kind for b in r.bursts]
    assert len(set(kinds[0::2])) == 1 and len(set(kinds[1::2])) == 1 and kinds[0] != kinds[1] # Each type its own kind
    assert np.median([b.evm for b in r.bursts]) < 0.2

def test_burst_kinds_when_the_gate_misses_by_tens_of_symbols():
    """A long burst's gate can open tens of symbols from its start: the preamble is still found (a wider alignment)"""
    from types import SimpleNamespace
    rng = np.random.default_rng(9)
    pts = np.exp(1j*np.pi/4*np.array([1, 3, 5, 7]))
    uw = pts[rng.integers(0, 4, 64)]
    B = []
    for i in range(8):
        lead = rng.integers(0, 25) # symbols of noise before the preamble: the gate opened that early
        s = np.concatenate((0.3*(rng.standard_normal(lead) + 1j*rng.standard_normal(lead)), uw, pts[rng.integers(0, 4, 4000)]))
        s = s*np.exp(1j*rng.uniform(0, 2*np.pi)) # each burst its own phase
        B.append(SimpleNamespace(symbols=s, start=0, stop=len(s)))
    kinds, member = burst.burst_kinds(B, 1.0, pts)
    assert len(kinds) == 1 and len(kinds[0]) >= 60 and member == [0]*8

def test_bursts_far_off_carrier():
    """Bursts 30 kHz off at 250 kBd (0.75 rad a symbol, too much for the preamble search's pieces): each burst's blind
    offset comes off first, so every one is acquired and its offset reported in full"""
    rng = np.random.default_rng(11)
    q = rp.symbol.load("qpsk").points
    fs, baud = 2e6, 250e3
    pre = q[rng.integers(0, 4, 48)]
    x = np.zeros(int(0.04*fs), complex)
    at, offs = 2000, []
    for i in range(8):
        syms = np.concatenate((pre, q[rng.integers(0, 4, 700)]))
        w = FR.word_waveform(syms, fs/baud)
        f = 30e3 + rng.uniform(-1e3, 1e3)
        w = w*np.exp(1j*rng.uniform(0, 2*np.pi) + 2j*np.pi*f/fs*np.arange(len(w)))
        x[at:at + len(w)] += w
        offs.append(f)
        at += len(w) + 800
    x += 0.05*(rng.standard_normal(len(x)) + 1j*rng.standard_normal(len(x)))
    r = burst.demodulate(x, fs, mod="qpsk", baud=baud, preamble=pre)
    assert len(r.bursts) == 8 and all(b.ok for b in r.bursts)
    assert all(abs(b.cfo - f) < 300 for b, f in zip(r.bursts, offs)) and np.median([b.evm for b in r.bursts]) < 0.2

def test_long_bursts_follow_their_symbol_clock():
    """3000-symbol bursts 200 ppm fast or slow (0.6 symbol over the burst, which a timing held from the preamble loses):
    each burst's clock is followed and measured, and its LLRs agree with its bits. Measured over 30 seeds: EVM <= 4.9 %,
    clock within 3.6 ppm"""
    rng = np.random.default_rng(51)
    q = rp.symbol.load("qpsk").points
    fs, baud = 1e6, 250e3
    pre = q[rng.integers(0, 4, 48)]
    x = np.zeros(int(4*(3000*fs/baud + 900) + 4000), complex)
    at, ppms = 2000, (200, -200, 200, -200)
    for ppm in ppms:
        syms = np.concatenate((pre, q[rng.integers(0, 4, 3000)]))
        w = FR.word_waveform(syms, fs/(baud*(1 + ppm*1e-6)))
        w = w*np.exp(1j*rng.uniform(0, 2*np.pi) + 2j*np.pi*rng.uniform(-2e3, 2e3)/fs*np.arange(len(w)))
        x[at:at + len(w)] += w
        at += len(w) + 900
    x += 0.05*(rng.standard_normal(len(x)) + 1j*rng.standard_normal(len(x)))
    r = burst.demodulate(x, fs, mod="qpsk", baud=baud, preamble=pre)
    assert len(r.bursts) == 4 and all(b.ok for b in r.bursts)
    for b, ppm in zip(r.bursts, ppms):
        assert b.evm < 0.1 and abs(1e6*b.clock - ppm) < 15
        assert np.array_equal(b.llr < 0, b.bits.astype(bool))
    L = r.llrs()
    assert len(L) == 4 and all(len(l) == 2*(len(b.symbols) - len(pre)) for l, b in zip(L, r.bursts))

def touching(rng, n=8, nsym=400, fs=2e6, baud=250e3, guard=4e-6, gains=(1.0,)):
    """n bursts from different transmitters (each its own carrier and phase) `guard` apart, no filter tails between
    them: (x, fs, baud, preamble, true starts)"""
    q = rp.symbol.load("qpsk").points
    pre = q[rng.integers(0, 4, 32)]
    x = np.zeros(int(n*(nsym + 60)*fs/baud + 20000), complex)
    at, starts = 5000, []
    for i in range(n):
        w = FR.word_waveform(np.concatenate((pre, q[rng.integers(0, 4, nsym)])), fs/baud)[int(8*fs/baud):-int(8*fs/baud)]
        g = gains[i % len(gains)]
        x[at:at + len(w)] += g*w*np.exp(1j*rng.uniform(0, 2*np.pi) + 2j*np.pi*rng.uniform(-2e3, 2e3)/fs*np.arange(len(w)))
        starts.append(at)
        at += len(w) + int(guard*fs)
    x += 0.04*(rng.standard_normal(len(x)) + 1j*rng.standard_normal(len(x)))
    return x, fs, baud, pre, starts

def test_touching_bursts_are_split():
    """Bursts a symbol apart (a guard the gate's hang merges), or with none at a power step between transmitters: cut
    where they meet. Measured over 20 seeds each: every start found to 40 us, one extra cut in 600"""
    for kw in (dict(guard=4e-6), dict(guard=0.0, gains=(1.0, 0.4)), dict(guard=3e-6, gains=(1.0, 0.5))):
        x, fs, baud, pre, starts = touching(np.random.default_rng(4), **kw)
        assert len(FR.gate(x, fs, touching=False)) < len(starts) # what the gate alone does: merged
        spans = FR.gate(x, fs)
        assert len(spans) == len(starts) and all(abs(a - s) < 40e-6*fs for (a, _), s in zip(spans, starts))
    x, fs, baud, pre, starts = touching(np.random.default_rng(5), n=1, nsym=4000) # one long burst stays whole
    assert len(FR.gate(x, fs)) == 1

def test_touching_bursts_are_demodulated():
    """Measured over 6 seeds: 47 of 48 bursts demodulated split, none merged"""
    x, fs, baud, pre, starts = touching(np.random.default_rng(2), gains=(1.0, 0.6))
    r = burst.demodulate(x, fs, mod="qpsk", baud=baud, preamble=pre)
    assert len(r.bursts) == 8 and sum(b.ok and b.evm < 0.15 for b in r.bursts) >= 7

def bursts_with(rng, layout, n=10, fs=2e6, baud=250e3):
    """n QPSK bursts, each its own carrier and phase: layout(word, rng) gives each burst's symbols around a shared word"""
    q = rp.symbol.load("qpsk").points
    word = q[rng.integers(0, 4, 26)]
    x = np.zeros(int(n*800*fs/baud), complex)
    at, sent = 3000, []
    for _ in range(n):
        syms = layout(word, rng, q)
        w = FR.word_waveform(syms, fs/baud)
        x[at:at + len(w)] += w*np.exp(1j*rng.uniform(0, 2*np.pi) + 2j*np.pi*rng.uniform(-2e3, 2e3)/fs*np.arange(len(w)))
        sent.append(syms)
        at += len(w) + 1500
    x += 0.05*(rng.standard_normal(len(x)) + 1j*rng.standard_normal(len(x)))
    return x, fs, q, sent

def symbol_errors(b, syms, q):
    """Fraction of b's symbols wrong against those sent, at the best shift (the gate's slack) and rotation"""
    best = 1.0
    for sh in range(-4, 5):
        a, s = b.symbols[max(0, sh):], syms[max(0, -sh):]
        m = min(len(a), len(s)) - 4
        dec = q[np.argmin(np.abs(a[:m, None] - q[None, :]), axis=1)]
        best = min(best, min(np.mean(np.abs(dec - s[:m]*1j**k) > 1e-6) for k in range(4)))
    return best

def test_known_symbols_in_the_middle_or_at_the_end():
    """A mid-amble (58 data symbols, 26 known, 300 data) and a postamble after payloads of several lengths: each burst
    acquired on them, and demodulated from its first symbol. Measured over 5 seeds each: 50 of 50 bursts without error"""
    mid = lambda w, r, q: np.concatenate((q[r.integers(0, 4, 58)], w, q[r.integers(0, 4, 300)]))
    post = lambda w, r, q: np.concatenate((q[r.integers(0, 4, int(r.integers(250, 450)))], w))
    for layout, where in ((mid, lambda o: 56 <= o <= 62), (post, lambda o: -30 <= o <= -26)):
        x, fs, q, sent = bursts_with(np.random.default_rng(1), layout)
        r = burst.demodulate(x, fs)
        assert not r.blind and len(r.preamble) >= 24 and where(r.offsets[0])
        assert len(r.bursts) == 10 and all(b.ok and symbol_errors(b, s, q) < 0.01 for b, s in zip(r.bursts, sent))
        a, n = r.bursts[0].known
        assert len(r.bursts[0].payload) == len(r.bursts[0].symbols) - n

def test_bursts_that_share_nothing_are_each_demodulated_blind():
    x, fs, q, sent = bursts_with(np.random.default_rng(2), lambda w, r, q: q[r.integers(0, 4, 384)])
    r = burst.demodulate(x, fs)
    assert r.blind and "blind" in str(r)
    assert len(r.bursts) == 10 and all(b.ok and symbol_errors(b, s, q) < 0.01 for b, s in zip(r.bursts, sent))

def test_slow_bursts_are_not_cut_by_their_own_envelope():
    """25 kBd: the symbols' envelope dips outlast the gate's default smoothing and hang, which scale with the occupied
    bandwidth instead. Measured: 0 of 20 bursts cut"""
    x, fs, baud, pre, starts = touching(np.random.default_rng(3), n=1, nsym=800, fs=200e3, baud=25e3)
    assert len(FR.gate(x, fs)) == 1

def test_long_bursts_of_frames_acquired_on_their_repeating_word():
    """Long QPSK bursts (12000 symbols, 4.6 samples a symbol) of 1482-symbol frames with a 10-symbol unique word, each
    burst at its own frame phase: not cut by their own envelope (the split scales with the bandwidth), and acquired
    on the word they repeat; the word decodes in every frame of every burst. (A real capture of this kind went blind)"""
    rng = np.random.default_rng(0)
    q = rp.symbol.load("qpsk").points
    Fs, baud, P = 781.25e3, 170.67e3, 1482
    uw = q[rng.integers(0, 4, 10)]
    xs = [np.zeros(8000, complex)]
    for _ in range(3):
        syms = q[rng.integers(0, 4, 12000)]
        for o in range(int(rng.integers(0, P)), 12000 - 10, P):
            syms[o:o + 10] = uw
        w = FR.word_waveform(syms, Fs/baud)
        xs += [w*np.exp(1j*rng.uniform(0, 2*np.pi) + 2j*np.pi*rng.uniform(-500, 500)/Fs*np.arange(len(w))), np.zeros(14000, complex)]
    x = np.concatenate(xs)
    x += 0.05*(rng.standard_normal(len(x)) + 1j*rng.standard_normal(len(x)))
    assert len(FR.gate(x, Fs)) == 3
    r = burst.demodulate(x, Fs, mod="qpsk", baud=baud)
    assert r.period == P and r.offsets == [None] and min(np.mean(np.abs(r.preamble*1j**k - uw) > 1e-6) for k in range(4)) == 0
    every = (len(xs[1]) + len(xs[2]))/Fs # one burst a period: the frame is that, not twice it
    assert abs(r.frame - every) < 1e-4
    for b in r.bursts:
        a, n = b.known
        dec = q[np.argmin(np.abs(b.symbols[:, None] - q[None, :]), axis=1)]
        assert all(np.array_equal(dec[o:o + n], r.preamble) for o in range(a, len(dec) - n, P))

def test_long_framed_bursts_take_their_carrier_from_every_word():
    """The same bursts 3 kHz off, with phase noise, at 12 dB: the carrier from one 9-symbol word drifts off over 12000
    symbols; from every word (the M-th power's offset over the burst, then each word's phase anchoring it) it holds.
    Measured over the synthetic real-capture case: EVM 20% (the SNR's), from 56% on one word"""
    rng = np.random.default_rng(3)
    q = rp.symbol.load("qpsk").points
    Fs, baud, P = 781.25e3, 170.67e3, 1482
    uw = q[rng.integers(0, 4, 9)]
    xs = [np.zeros(8000, complex)]
    for _ in range(3):
        syms = q[rng.integers(0, 4, 12000)]
        for o in range(int(rng.integers(0, P)), 12000 - 9, P):
            syms[o:o + 9] = uw
        w = FR.word_waveform(syms, Fs/baud)
        m = np.arange(len(w))
        w = w*np.exp(1j*(rng.uniform(0, 2*np.pi) + 2*np.pi*rng.uniform(-3e3, 3e3)/Fs*m + np.cumsum(2e-3*rng.standard_normal(len(w)))))
        xs += [w, np.zeros(14000, complex)]
    x = np.concatenate(xs)
    x += np.sqrt(np.mean(np.abs(xs[1])**2)/10**1.2/2)*(rng.standard_normal(len(x)) + 1j*rng.standard_normal(len(x)))
    r = burst.demodulate(x, Fs, mod="qpsk", baud=baud)
    assert r.period == P and len(r.bursts) == 3 and all(b.ok and b.evm < 0.3 for b in r.bursts)

def test_weak_sparse_bursts_with_a_long_preamble():
    """BPSK bursts at 4 dB over the floor, sparse, a 652-symbol 1100 preamble: the gate does not fire on noise (levels
    over 64 symbols), the rate is the line near the occupied bandwidth (the preamble's own lines at a half and 2x lie
    outside it), and the preamble is found though decision errors break single steps"""
    rng = np.random.default_rng(0)
    Fs, baud = 390.625e3, 19.2e3
    pre = np.tile([1, 1, -1, -1], 163).astype(complex)
    xs = [np.zeros(int(0.2*Fs), complex)]
    for _ in range(5):
        syms = np.concatenate((pre, np.array([1, -1], complex)[rng.integers(0, 2, 1710)]))
        w = FR.word_waveform(syms, Fs/baud)
        xs += [w*np.exp(1j*rng.uniform(0, 2*np.pi) + 2j*np.pi*rng.uniform(-100, 100)/Fs*np.arange(len(w))), np.zeros(int(0.5*Fs), complex)]
    x = np.concatenate(xs)
    p = np.mean(np.abs(x[np.abs(x) > 1e-9])**2)
    x += np.sqrt(p/10**0.4/2)*(rng.standard_normal(len(x)) + 1j*rng.standard_normal(len(x)))
    assert len(FR.gate(x, Fs)) == 5
    assert FR.gate(np.sqrt(p/2)*(rng.standard_normal(len(x)) + 1j*rng.standard_normal(len(x))), Fs) == [] # noise alone
    r = burst.demodulate(x, Fs)
    assert r.mod == "bpsk" and abs(r.baud - baud) < 100 and not r.blind and len(r.preamble) >= 600
    assert len(r.bursts) == 5 and all(b.ok for b in r.bursts)
    assert all(abs(b.clock) < 50e-6 for b in r.bursts) # (the 1100 preamble's energy peaks between symbols: not a clock)

def test_four_weak_bursts_share_their_preamble():
    """Four bursts at 2 dB over the floor: a decision error in one burst must not break the shared run (all but one
    agree), and the run is still longer than chance could make for BPSK's two step values"""
    rng = np.random.default_rng(5)
    Fs, baud = 390.625e3, 19.2e3
    pre = np.tile([1, 1, -1, -1], 163).astype(complex)
    xs = [np.zeros(int(0.2*Fs), complex)]
    for _ in range(4):
        syms = np.concatenate((pre, np.array([1, -1], complex)[rng.integers(0, 2, 1700)]))
        w = FR.word_waveform(syms, Fs/baud)
        xs += [w*np.exp(1j*rng.uniform(0, 2*np.pi)), np.zeros(int(0.5*Fs), complex)]
    x = np.concatenate(xs)
    p = np.mean(np.abs(xs[1])**2)
    x += np.sqrt(p/10**0.2/2)*(rng.standard_normal(len(x)) + 1j*rng.standard_normal(len(x)))
    r = burst.demodulate(x, Fs, mod="bpsk", baud=baud)
    assert not r.blind and len(r.preamble) >= 600 and len(r.bursts) == 4 and all(b.ok for b in r.bursts)
