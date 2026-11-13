"""projects/adsb/adsb.py on synthetic captures: published Mode S test messages, pulse-position modulated at 2 MS/s with noise and an unknown carrier phase

The messages are the widely used worked examples (identification KLM1023; an even/odd airborne position pair, 38000 ft near 52.2572 N 3.91937 E)."""
import pathlib
import sys

import numpy as np
import pytest

pytest.importorskip("pysdrlib")
PROJECTS = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECTS))
import _paths  # noqa: E402,F401
import adsb  # noqa: E402

IDENT = "8D4840D6202CC371C32CE0576098"
EVEN = "8D40621D58C382D690C8AC2863A7"
ODD = "8D40621D58C386435CC412692AD6"

def modulate(msgs, gap=400, lead=300, amp=1.0, phase=0.7):
    """Mode S 1090 MHz frames as a 2 MS/s envelope (preamble 0/1/3.5/4.5 us, then 112 bits of 1 us, a 1 = first half high), as complex samples"""
    out = np.zeros(lead + len(msgs)*(240 + gap), np.complex64)
    for k, h in enumerate(msgs):
        i = lead + k*(240 + gap)
        for p in (0, 2, 7, 9):
            out[i + p] = amp
        bits = np.unpackbits(np.frombuffer(bytes.fromhex(h), np.uint8))
        out[i + 16:i + 16 + 224:2] = amp*bits
        out[i + 17:i + 17 + 224:2] = amp*(1 - bits)
    return out*np.exp(1j*phase)

def noisy(x, snr_db, rng):
    sig = np.sqrt(np.mean(np.abs(x[np.abs(x) > 0])**2))
    s = sig*10**(-snr_db/20)/np.sqrt(2)
    return x + (rng.normal(0, s, len(x)) + 1j*rng.normal(0, s, len(x))).astype(np.complex64)

def test_messages_found_in_order(rng):
    x = noisy(modulate([IDENT, EVEN, ODD]), 25, rng)
    got = [b.hex().upper() for _, b in adsb.messages(x)]
    assert got == [IDENT, EVEN, ODD]

def test_decode_aircraft(rng):
    x = noisy(modulate([IDENT, EVEN, ODD]), 25, rng)
    n, ac = adsb.decode(x)
    assert n == 3
    assert ac["4840d6"].call == "KLM1023"
    a = ac["40621d"]
    assert a.alt == 38000
    assert a.pos == pytest.approx((52.26578, 3.93891), abs=2e-4) # the odd frame is the newer: its own latitude

def test_position_uses_newer_frame(rng):
    n, ac = adsb.decode(noisy(modulate([ODD, EVEN]), 25, rng))
    assert ac["40621d"].pos == pytest.approx((52.25720, 3.91937), abs=2e-4)

def test_noise_only_gives_nothing(rng):
    x = (rng.normal(size=20000) + 1j*rng.normal(size=20000)).astype(np.complex64)
    assert adsb.decode(x)[0] == 0

def test_corrupt_message_rejected(rng):
    bad = IDENT[:-4] + "0000"
    n, _ = adsb.decode(noisy(modulate([bad, IDENT]), 25, rng))
    assert n == 1 # the CRC check drops the damaged frame, keeps the good one

def test_cpr_needs_both_frames():
    ac = adsb.Aircraft()
    ac.update(bytes.fromhex(EVEN), 0.0)
    assert ac.pos is None
