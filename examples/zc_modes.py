"""A preamble that says what follows: Zadoff-Chu roots pick each frame's modulation and code

    python examples/zc_modes.py

The transmitter starts every frame with a Zadoff-Chu sequence of length 63 whose root names the frame's mode, from a
table: root 25 QPSK with Hamming (7,4), root 29 8PSK with Hamming (7,4), root 34 16APSK uncoded. The receiver's frame
sync correlates all three roots, and the one it finds is published with the frame (`word`, `root`, `shift`, and what
the preamble measured: `cfo`, `esn0`, `scale`), so `controls` drive the constellation and the code from the same table,
frame by frame. The `symbol marker` step puts the root's symbols before each frame on transmit and strips them on
receive. All of it is definitions: a table, a profile.
"""
import json
import pathlib
import tempfile

import numpy as np

import rpps as rp

N, SYMBOLS = 63, 420 # Preamble length; symbols of a frame after it (a whole number of Hamming words in every mode)
MODES = [[25, "psk/qpsk", "hamming/7_4", SYMBOLS*2*4//7],
         [29, "psk/8psk", "hamming/7_4", SYMBOLS*3*4//7],
         [34, "apsk/dvbs2_16apsk_3_4", None, SYMBOLS*4]]

DEFINITIONS = {
    "table/demo/zc_modes": {
        "base": "table", "comment": "Zadoff-Chu root -> the frame's constellation, code and data bits",
        "key": "root", "columns": ["root", "constellation", "code", "bits"], "rows": MODES},
    "profile/demo/zc_modes": {
        "base": "profile",
        "comment": "Demo: frames of 420 symbols, each after a 63-symbol Zadoff-Chu preamble whose root says its "
                   "modulation and code (table demo/zc_modes). The input is symbols, one a sample",
        "steps": [
            {"sync": "frame", "id": "fs", "zc": {"roots": [25, 29, 34], "length": N}, "length": N + SYMBOLS,
             "segments": 2, "cfo": True, "threshold": 0.5},
            {"symbol": "marker", "id": "pre", "zc": {"length": N, "root": 25}},
            {"symbol": "constellation", "id": "map", "name": "psk/qpsk"},
            {"code": "block", "id": "fec", "name": "hamming/7_4"}],
        "controls": {
            "fs": {"table": "demo/zc_modes", "key": "root",
                   "drive": {"pre": {"frame": "frame"}, "map": {"name": "constellation"}, "fec": {"name": "code"}}},
            "pre": {"on": "tx", "table": "demo/zc_modes", "key": "root", "frame": "bits",
                    "schedule": [{"root": 25}, {"root": 29}, {"root": 34}],
                    "drive": {"pre": {"root": "root"}, "map": {"name": "constellation"}, "fec": {"name": "code"}}}}},
}

def main():
    d = pathlib.Path(tempfile.mkdtemp())
    for ref, obj in DEFINITIONS.items():
        f = d/f"{ref}.json"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(obj, indent=1))
    rp.add_config_path(d)

    p = rp.profile.load("demo/zc_modes")
    rng = np.random.default_rng(0)
    per = [m[3] for m in MODES]*3 # nine frames: the schedule three times over
    data = rng.integers(0, 2, sum(per)).astype(bool)
    sym = np.asarray(p.tx()(rp.Data(data, sps=1, flush=True)).data)
    assert len(sym) == 9*(N + SYMBOLS)

    # The channel: a carrier offset and phase, noise at 26 dB Es/N0 (the 16APSK frames are uncoded), some noise before the first frame
    n = np.arange(len(sym))
    y = sym*np.exp(2j*np.pi*0.001*n + 1.1j)
    y = np.concatenate(((rng.normal(size=300) + 1j*rng.normal(size=300))*0.05, y))
    y = (y + (rng.normal(size=len(y)) + 1j*rng.normal(size=len(y)))*np.sqrt(10**(-26/10)/2)).astype(np.complex64)

    rx = p.rx()
    out, stream = [], None
    for blk in rp.source.Memory(y, Fs=1.0, sps=1)(1000):
        r = blk + rx
        stream = r.meta.stream
        out.append(np.asarray(r.bits()) if len(r) else np.zeros(0, bool))
    found = [f for _, f, _ in stream.control.fields("fs")]
    got = np.concatenate(out)[:len(data)]
    roots = [f["root"] for f in found]
    print(f"{len(found)} frames, roots {roots}, cfo {np.median([f['cfo'] for f in found]):.4f} cycles a symbol (0.001 sent), "
          f"Es/N0 {np.median([f['esn0'] for f in found]):.1f} dB (26 sent); {np.count_nonzero(got != data)} bit errors")
    assert roots == [25, 29, 34]*3
    assert np.array_equal(got, data)

if __name__ == "__main__":
    main()
