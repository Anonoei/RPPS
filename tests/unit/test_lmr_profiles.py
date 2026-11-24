"""P25 phase 1 TSDUs and DMR bursts as profiles (profile/p25/tsdu, profile/dmr/burst): sync words among the
discriminator's levels, status symbols, codes split around a sync word, branches that read part of each frame. Their
receive sides end in branches (receive only), so the signals are built here from the same definitions"""
import numpy as np
import pytest

import rpps as rp

def _bits(v, n):
    return np.array([(v >> (n - 1 - i)) & 1 for i in range(n)], bool)

def _fm(levels, h, rng, noise=0.05):
    tx = rp._config.load_obj("profile", "test/fm4", {"base": "profile", "params": {"sps": {"default": 10}}, "steps": [
        {"wave": "fm", "h": h, "sps": {"param": "sps"}}, {"wave": "shape", "pulse": "rc/0p20_frequency", "sps": {"param": "sps"}}]}).tx()
    x = np.asarray(tx(rp.Data(levels.astype(np.float32), Fs=4800, sps=1, flush=True)).data)
    return (x*np.exp(0.4j) + (rng.normal(size=len(x)) + 1j*rng.normal(size=len(x)))*noise).astype(np.complex64)

def _lead(rng, n=300):
    """Random levels before (and after) the frames: the timing loop locks on them, as on a channel that was on already"""
    return rng.choice([-3, -1, 1, 3], n)/np.sqrt(5)

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
        body = [fs, nid.encode(_bits(nac << 4 | 7, 16))]
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
    return np.concatenate([_lead(rng)] + frames + [_lead(rng, 60)]), sent

@pytest.mark.parametrize("block", [4096, 1777])
@pytest.mark.parametrize("counts, fill", [((3, 3, 3, 3, 3), False), ((1, 2, 3, 1, 1, 2), False), ((2, 1, 3, 2, 1), True)])
def test_p25_tsdu(rng, block, counts, fill):
    """TSDUs of one to three TSBKs, read to the next frame sync: each TSBK out once, its CRC good"""
    levels, sent = p25_tsdus(rng, counts, fill=fill)
    msgs = _receive("p25/tsdu", _fm(levels, 0.25, rng), block)
    nid = [m for m in msgs if m.kind == "nid"]
    tsbk = [m for m in msgs if m.kind == "tsbk"]
    assert [(m.fields["nac"], m.fields["duid.name"]) for m in nid] == [(0x293, "TSDU")]*len(counts)
    assert all(m.crc_ok for m in tsbk)
    got = [bytes([m.fields["lb"] << 7 | m.fields["p"] << 6 | m.fields["opcode"], m.fields["mfid"]]) + bytes(m.fields["args"])
           for m in tsbk]
    assert got == [s[:10] for s in sent]

def dmr_bursts(rng, n=16):
    bptc, itl = rp.code.load("block/product/dmr_bptc").encoder, rp.code.load("interleave/block/dmr_bptc").encoder
    gol, con = rp.code.load("block/golay/20_8").encoder, rp.symbol.load("ask/4level")
    words = {w: np.asarray(rp.sequence.load(f"dmr_{w}").bits, bool) for w in ("bs_voice", "bs_data")}
    slots, sent = [], []
    for k in range(n):
        kind = "bs_data" if k % 3 else "bs_voice"
        dt = 3 if kind == "bs_data" else 1
        st = gol.encode(_bits(5 << 4 | dt, 8))
        info = rng.integers(0, 2, 96).astype(bool)
        c = itl.encode(bptc.encode(info)) if kind == "bs_data" else rng.integers(0, 2, 196).astype(bool)
        burst = np.concatenate((c[:98], st[:10], words[kind], st[10:], c[98:]))
        slots.append(np.concatenate((rng.integers(0, 2, 24).astype(bool), burst))) # (the CACH, then the burst)
        sent.append((dt, info if kind == "bs_data" else None))
    sym = np.asarray((rp.Data(np.concatenate(slots)) * con).data).real
    return np.concatenate((_lead(rng), sym, _lead(rng, 60))), sent

@pytest.mark.parametrize("block", [3000, 1111])
def test_dmr_bursts(rng, block):
    """Every burst's slot type, across the sync word; only the data bursts' BPTC payloads (select by the sync word)"""
    levels, sent = dmr_bursts(rng)
    msgs = _receive("dmr/burst", _fm(levels, 0.27, rng), block)
    st = [m for m in msgs if m.kind == "slot_type"]
    info = [m for m in msgs if m.kind == "info"]
    assert [(m.fields["cc"], m.fields["dt"]) for m in st] == [(5, dt) for dt, _ in sent]
    want = [i for _, i in sent if i is not None]
    assert [np.unpackbits(np.frombuffer(bytes(m.fields["info"]), np.uint8)).astype(bool).tolist() for m in info] == \
           [w.tolist() for w in want]
