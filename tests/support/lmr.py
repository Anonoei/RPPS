"""4-level land mobile radio signals (DMR, P25), built from the shipped definitions"""
import numpy as np

import rpps as rp

def field_bits(v, n):
    return np.array([(v >> (n - 1 - i)) & 1 for i in range(n)], bool)

def fm4(levels, h, rng, noise=0.05):
    tx = rp._config.load_obj("profile", "test/fm4", {"base": "profile", "params": {"sps": {"default": 10}}, "steps": [
        {"wave": "fm", "h": h, "sps": {"param": "sps"}}, {"wave": "shape", "pulse": "rc/0p20_frequency", "sps": {"param": "sps"}}]}).tx()
    x = np.asarray(tx(rp.Data(levels.astype(np.float32), Fs=4800, sps=1, flush=True)).data)
    return (x*np.exp(0.4j) + (rng.normal(size=len(x)) + 1j*rng.normal(size=len(x)))*noise).astype(np.complex64)

def lead_levels(rng, n=300):
    """Random levels before (and after) the frames: the timing loop locks on them, as on a channel that was on already"""
    return rng.choice([-3, -1, 1, 3], n)/np.sqrt(5)

def dmr_bursts(rng, n=16):
    bptc, itl = rp.code.load("block/product/dmr_bptc").encoder, rp.code.load("interleave/block/dmr_bptc").encoder
    gol, con = rp.code.load("block/golay/20_8").encoder, rp.symbol.load("ask/4level")
    words = {w: np.asarray(rp.sequence.load(f"dmr_{w}").bits, bool) for w in ("bs_voice", "bs_data")}
    slots, sent = [], []
    for k in range(n):
        kind = "bs_data" if k % 3 else "bs_voice"
        dt = 3 if kind == "bs_data" else 1
        st = gol.encode(field_bits(5 << 4 | dt, 8))
        info = rng.integers(0, 2, 96).astype(bool)
        c = itl.encode(bptc.encode(info)) if kind == "bs_data" else rng.integers(0, 2, 196).astype(bool)
        burst = np.concatenate((c[:98], st[:10], words[kind], st[10:], c[98:]))
        slots.append(np.concatenate((rng.integers(0, 2, 24).astype(bool), burst))) # (the CACH, then the burst)
        sent.append((dt, info if kind == "bs_data" else None))
    sym = np.asarray((rp.Data(np.concatenate(slots)) * con).data).real
    return np.concatenate((lead_levels(rng), sym, lead_levels(rng, 60))), sent
