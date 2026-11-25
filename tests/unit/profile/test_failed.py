"""A frame the decoders could not decode is dropped, whatever its header's CRC says (the `failed` mask, rpps.code.Coding)"""
import numpy as np

import rpps as rp
from support.dvbs2 import KNOWN, MIXED_FRAMES as FRAMES, seeded_bits as bits


def received(noise, parallel=None):
    p = rp.profile.load("dvbs2/normal", sync=KNOWN)
    t = rp.profile.table("dvbs2/modcod")
    dfl = [t.row({**f, "short": 0})["kbch"] - 80 for f in FRAMES]
    data = bits(sum(dfl), 4)
    o = p.tx()(rp.Data(data, Fs=1e6, sps=1, flush=True), fields=FRAMES)
    x = np.concatenate((o.data, np.zeros(200, np.complex64)))
    rng = np.random.default_rng(5)
    y = (x*np.exp(0.7j) + noise*(rng.normal(size=len(x)) + 1j*rng.normal(size=len(x)))).astype(np.complex64)
    rx = p.rx() if parallel is None else p.rx(parallel=parallel)
    out = rx(rp.Data(y, Fs=o.Fs, sps=4, delay=o.delay, flush=True))
    got = np.asarray(out.data).astype(bool)
    if hasattr(rx, "close"):
        rx.close()
    return got, data, dfl, out.stream

def test_a_frame_whose_code_failed_is_dropped():
    """About -3 dB Es/N0: LDPC and BCH fail on every frame. A header can still pass its 8-bit CRC (the code under it got
    those bits right): the frame goes anyway, counted"""
    got, data, dfl, st = received(0.5)
    assert len(got) == 0
    est = st.est["dvbs2_bb"]
    assert est["frames"] + est["crc_failed"] + est["fec_failed"] == 3 and est["frames"] == 0

def test_failed_frames_are_dropped_by_the_parallel_runner_too():
    serial, *_ = received(0.5)
    parallel, *_ = received(0.5, rp.parallel.Parallel("inline"))
    assert len(serial) == len(parallel) == 0

def test_clean_frames_carry_no_mask():
    got, data, dfl, st = received(0.0)
    assert np.array_equal(got, data) and st.est["dvbs2_bb"]["fec_failed"] == 0
