"""Your own signal as definitions, not code

    python examples/own_definitions.py

Everything RPPS can load (constellations, pulses, sequences, scramblers, codes, headers, tables, profiles) is a JSON
definition, found by name in config/ and in the directories you add with `rp.add_config_path(dir)` (or `RPPS_CONFIG`),
laid out the same way. A new signal is usually a profile that strings shipped definitions together, plus the one or
two things that are really yours: here a 32-bit sync word. A definition can `extend` another and change only what
differs. Nothing in the package is touched, and the same files load from the shell (`python -m rpps rx ...`).

Here: a made-up beacon. BPSK with RRC pulses; frames of `payload` bytes, each Hamming (7,4) coded, randomized with the
CCSDS sequence and sent behind our own sync word. Timing and carrier come from the block estimators; the 180 degree
ambiguity of BPSK is put right by the sync word's polarity. A second profile extends it with longer frames.
"""
import json
import pathlib
import tempfile

import numpy as np

import rpps as rp

FRAME = "7*8*payload//4" # channel bits a frame: payload bytes, Hamming (7,4) coded

DEFINITIONS = {
    "sequence/word/beacon_sync": {
        "base": "sequence", "type": "word", "comment": "The beacon's sync word, most significant bit first",
        "word": "0xF3A2B5C7", "bits": 32, "order": "msb"},
    "profile/demo/beacon": {
        "base": "profile",
        "comment": "Demo beacon: BPSK, RRC 0.35, frames of `payload` bytes, Hamming (7,4), randomized, behind a sync word",
        "params": {"sps": {"default": 4, "comment": "samples per symbol"},
                   "payload": {"default": 32, "comment": "data bytes a frame"}},
        "steps": [
            {"wave": "shape", "pulse": "rrc/0p35", "sps": {"param": "sps"}},
            {"sync": "timing"},
            {"sync": "carrier", "order": 2, "constellation": "bpsk"},
            {"symbol": "constellation", "name": "bpsk"},
            {"sync": "delimit", "id": "sync", "sequence": "beacon_sync", "length": {"expr": FRAME},
             "keep": True, "polarity": True, "errors": 2},
            {"code": "marker", "sequence": "beacon_sync", "length": {"expr": FRAME}},
            {"code": "scramble", "name": "additive/ccsds", "length": {"expr": FRAME}},
            {"code": "block", "name": "hamming/7_4"}]},
    "profile/demo/beacon_long": { # only what differs from its parent
        "base": "profile", "extends": "demo/beacon", "comment": "Demo beacon with 64-byte frames",
        "params": {"payload": 64}},
}

def loopback(name, nframes=6, esn0_db=8.0):
    p = rp.profile.load(name)
    payload = p.params["payload"]
    rng = np.random.default_rng(0)
    data = bytes(rng.integers(0, 256, nframes*payload).astype(np.uint8))
    wave = p.tx()(rp.Data(data, Fs=9600, sps=1, flush=True))
    air = (wave + rp.signal.channel.CFO(30) + rp.signal.channel.PhaseOffset(2.0)
           + rp.signal.channel.AWGN(esn0_db=esn0_db, power=None, seed=1))
    x = np.concatenate([air.data, np.zeros(2000, np.complex64)]) # the receiver's filters run a little behind

    rx, got = p.rx(), []
    for blk in rp.source.Memory(x, Fs=wave.Fs, sps=wave.sps)(4096):
        out = rx(blk)
        stream = out.stream
        if len(out):
            got.append(bytes.fromhex(out.as_bytes().hex))
    got = b"".join(got)
    print(f"{p}\n  {p.params}: {len(data)} bytes sent, {len(got)} received, sync {stream.est['sync']}")
    assert got == data
    assert stream.est["sync"]["frames"] == nframes

def main():
    # Your own directory would normally live next to your code; a temporary one keeps this script self-contained
    with tempfile.TemporaryDirectory(prefix="rpps_config_") as tmp:
        root = pathlib.Path(tmp)
        for rel, obj in DEFINITIONS.items():
            f = root/f"{rel}.json"
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(json.dumps(obj, indent=2))
        rp.add_config_path(root) # searched before the shipped config; `python -m rpps ls` now lists these too
        print(f"definitions in {root}: {', '.join(DEFINITIONS)}")

        loopback("demo/beacon")
        loopback("demo/beacon_long")

if __name__ == "__main__":
    main()
