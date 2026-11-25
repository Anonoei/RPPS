"""P25 phase 1 TSDUs and DMR bursts as profiles (profile/p25/tsdu, profile/dmr/burst): sync words among the
discriminator's levels, status symbols, codes split around a sync word, branches that read part of each frame. Their
receive sides end in branches (receive only), so the signals are built here from the same definitions"""
import numpy as np
import pytest

import rpps as rp
from support.lmr import dmr_bursts, field_bits, fm4, lead_levels

def _receive(name, x, block):
    rx, msgs = rp.profile.load(name).rx(), []
    for blk in rp.source.Memory(x, Fs=48000, sps=10)(block):
        blk + rx
        msgs += rx.take_messages()
    return msgs

def p25_tsdus(rng, counts=(3, 3, 3, 3, 3), nac=0x293, fill=False):
    """TSDUs of `counts` TSBKs each; fill: null bits after the last up to the next status symbol"""
    crc = rp.link.load("crc/crc16_gsm").encoder
    nid, tre = rp.code.load("block/bch/p25_nid").encoder, rp.code.load("block/trellis/p25_1_2").encoder
    itl, con = rp.code.load("interleave/block/p25_tsbk").encoder, rp.symbol.load("ask/4level")
    fs = np.asarray(rp.sequence.load("p25_fs").bits, bool)
    frames, sent = [], []
    for count in counts:
        body = [fs, nid.encode(field_bits(nac << 4 | 7, 16))]
        for b in range(count):
            m = rng.integers(0, 256, 10).astype(np.uint8)
            m[0] = (m[0] & 0x3F) | (0x80 if b == count - 1 else 0) # the last block flag
            block = bytes(m) + crc.compute(bytes(m)).to_bytes(2, "big")
            sent.append(block)
            body.append(itl.encode(tre.encode(np.unpackbits(np.frombuffer(block, np.uint8)).astype(bool))))
        bits = np.concatenate(body)
        if fill:
            bits = np.concatenate((bits, np.zeros(-len(bits) % 70, bool)))
        sym = np.asarray((rp.Data(bits) * con).data).real
        frames.append(np.insert(sym, np.arange(35, len(sym) + 1, 35), -1/np.sqrt(5))) # a status symbol after every 35
    return np.concatenate([lead_levels(rng)] + frames + [lead_levels(rng, 60)]), sent

@pytest.mark.parametrize("block", [4096, 1777])
@pytest.mark.parametrize("counts, fill", [((3, 3, 3, 3, 3), False), ((1, 2, 3, 1, 1, 2), False), ((2, 1, 3, 2, 1), True)])
def test_p25_tsdu(rng, block, counts, fill):
    """TSDUs of one to three TSBKs, read to the next frame sync: each TSBK out once, its CRC good"""
    levels, sent = p25_tsdus(rng, counts, fill=fill)
    msgs = _receive("p25/tsdu", fm4(levels, 0.25, rng), block)
    nid = [m for m in msgs if m.kind == "nid"]
    tsbk = [m for m in msgs if m.kind == "tsbk"]
    assert [(m.fields["nac"], m.fields["duid.name"]) for m in nid] == [(0x293, "TSDU")]*len(counts)
    assert all(m.crc_ok for m in tsbk)
    got = [bytes([m.fields["lb"] << 7 | m.fields["p"] << 6 | m.fields["opcode"], m.fields["mfid"]]) + bytes(m.fields["args"])
           for m in tsbk]
    assert got == [s[:10] for s in sent]

@pytest.mark.parametrize("block", [3000, 1111])
def test_dmr_bursts(rng, block):
    """Every burst's slot type, across the sync word; only the data bursts' BPTC payloads (select by the sync word)"""
    levels, sent = dmr_bursts(rng)
    msgs = _receive("dmr/burst", fm4(levels, 0.27, rng), block)
    st = [m for m in msgs if m.kind == "slot_type"]
    info = [m for m in msgs if m.kind == "info"]
    assert [(m.fields["cc"], m.fields["dt"]) for m in st] == [(5, dt) for dt, _ in sent]
    want = [i for _, i in sent if i is not None]
    assert [np.unpackbits(np.frombuffer(bytes(m.fields["info"]), np.uint8)).astype(bool).tolist() for m in info] == \
           [w.tolist() for w in want]
