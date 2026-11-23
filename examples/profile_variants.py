"""Variants of one profile, and a frame check built from CRCs: still definitions, not code

    python examples/profile_variants.py

A parameter can list its "choices", each a bundle of fields that one choice sets together: `modulation` picks the
constellation the carrier, frame sync and symbol steps all name, and how sure the frame sync must be; `rolloff` picks
the pulse. A step names a field as {"param": "modulation.shape"} (or `modulation.shape` in an expression), and a step
with "when" is in the chain only where its expression holds (`fec == 'hamming'`). A value that is not a choice is an
error naming the choices.

A check that is not one CRC is a "composite" crc definition: CRC parts over transformed views of the message,
concatenated (or XORed). Here an HDLC FCS-16 whose first byte is the CRC-8 of the frame and whose second is the CRC-8
of the frame with each byte's nibbles swapped. It goes wherever a CRC does: a frame's `fcs`, a link header's `crc`.
"""
import json
import pathlib
import tempfile

import numpy as np

import rpps as rp

FRAME = "8*payload*7//4 if fec == 'hamming' else 8*payload" # channel bits a frame

DEFINITIONS = {
    "sequence/word/variant_sync": {
        "base": "sequence", "type": "word", "comment": "A 64-bit sync word, most significant bit first",
        "word": "0xF3A2B5C71D9E4A63", "bits": 64, "order": "msb"},
    "profile/demo/variants": {
        "base": "profile",
        "comment": "Demo: QPSK or 16QAM, RRC 0.2 or 0.35, frames of `payload` bytes behind a sync word, randomized, "
                   "Hamming (7,4) or uncoded",
        "params": {
            "sps": {"default": 4, "comment": "samples per symbol"},
            "payload": {"default": 32, "type": "int", "min": 1, "comment": "data bytes a frame"},
            "modulation": {"default": "qpsk", "comment": "the constellation, and how well the sync word must match",
                           "choices": {"qpsk": {"shape": "QPSK", "threshold": 0.6},
                                       "16qam": {"shape": "16QAM", "threshold": 0.85}}},
            "rolloff": {"default": 0.35, "comment": "the RRC roll-off",
                        "choices": {"0.2": {"pulse": "rrc/0p20"}, "0.35": {"pulse": "rrc/0p35"}}},
            "fec": {"default": "hamming", "choices": ["hamming", "none"]}},
        "requires": {"baud": None, "Fs": "baud*sps", "bw": "(1 + rolloff)*baud", "min_samples": "sps*4096"},
        "steps": [
            {"wave": "shape", "pulse": {"param": "rolloff.pulse"}, "sps": {"param": "sps"}},
            {"sync": "timing"},
            {"sync": "carrier", "order": 4, "shape": {"param": "modulation.shape"}},
            {"sync": "frame", "sequence": "variant_sync", "threshold": {"param": "modulation.threshold"}},
            {"symbol": "constellation", "shape": {"param": "modulation.shape"}},
            {"code": "marker", "sequence": "variant_sync", "length": {"expr": FRAME}},
            {"code": "scramble", "name": "additive/ccsds", "length": {"expr": FRAME}},
            {"code": "block", "name": "hamming/7_4", "when": "fec == 'hamming'"}]},
    "crc/demo_fcs16": {
        "base": "crc", "type": "composite",
        "comment": "FCS-16: the CRC-8 of the frame, then the CRC-8 of the frame with its bytes' nibbles swapped",
        "parts": [{"crc": "crc8_smbus"}, {"crc": "crc8_smbus", "input": ["nibble_swap"]}]},
    "frame/demo_hdlc": {"base": "frame", "extends": "hdlc", "comment": "HDLC with the demo FCS-16", "fcs": "demo_fcs16"},
}

def loopback(nframes=40, esn0_db=20.0, **variant):
    p = rp.profile.load("demo/variants", **variant)
    payload = p.params["payload"]
    rng = np.random.default_rng(0)
    data = bytes(rng.integers(0, 256, nframes*payload).astype(np.uint8))
    wave = p.tx()(rp.Data(data, Fs=9600, sps=1, flush=True))
    air = (wave + rp.signal.channel.CFO(30) + rp.signal.channel.PhaseOffset(2.0)
           + rp.signal.channel.AWGN(esn0_db=esn0_db, power=None, seed=1))
    x = np.concatenate([air.data, np.zeros(2000, np.complex64)]) # the receiver's filters run a little behind
    rx, got = p.rx(), b""
    for blk in rp.source.Memory(x, Fs=wave.Fs, sps=wave.sps)(4096):
        out = blk + rx
        got += bytes.fromhex(out.as_bytes().hex) if len(out) else b""
    info = rp.profile.info("demo/variants", baud=wave.Fs/wave.sps, **variant)
    print(f"{variant or 'defaults'}: {len(wave)} samples ({info.requires['bw']:g} Hz wide), "
          f"{len(data)} bytes {'back' if got[:len(data)] == data else 'NOT back'}")
    assert got[:len(data)] == data
    return len(wave)

def composite_fcs():
    fcs = rp.link.load("crc/demo_fcs16").encoder
    crc8 = rp.link.load("crc/crc8_smbus").encoder
    frame = bytes(range(1, 21))
    swapped = bytes(((b << 4) | (b >> 4)) & 0xFF for b in frame)
    assert fcs.to_bytes(fcs.compute(frame)) == bytes([crc8.compute(frame), crc8.compute(swapped)])

    hdlc = rp.link.load("frame/demo_hdlc")
    payloads = [bytes(np.random.default_rng(n).integers(0, 256, n).astype(np.uint8)) for n in (8, 30, 100)]
    bits = np.concatenate([(rp.Data(pl) * hdlc).data for pl in payloads])
    back = (rp.Data(bits, flush=True) / hdlc).as_bytes()
    assert bytes.fromhex(back.hex) == b"".join(payloads)
    bad = bits.copy()
    bad[40] ^= True # a bit of the first frame flipped: its FCS fails, the others come out
    assert bytes.fromhex((rp.Data(bad, flush=True) / hdlc).as_bytes().hex) == b"".join(payloads[1:])
    print(f"composite FCS-16 of {frame.hex()}: {fcs.to_bytes(fcs.compute(frame)).hex()}; "
          f"{len(payloads)} HDLC frames through it, a damaged one dropped")

def main():
    d = pathlib.Path(tempfile.mkdtemp())
    for ref, obj in DEFINITIONS.items():
        f = d/f"{ref}.json"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(obj, indent=1))
    rp.add_config_path(d)

    qpsk = loopback()
    qam = loopback(modulation="16qam", rolloff=0.2)
    assert abs(2*qam - qpsk) < 0.01*qpsk # twice the bits a symbol: half the samples (but for the filters' tails)
    loopback(fec="none")
    try:
        rp.profile.load("demo/variants", modulation="8psk")
        raise AssertionError("8psk is not a choice")
    except rp.errors.ConfigError as e:
        print(f"refused: {e}")
    composite_fcs()

if __name__ == "__main__":
    main()
