"""Diagnose an ATSC decode that doesn't work, and discover its parameters from the capture itself

    python projects/atsc/atsc_diag.py relation  --soft data/atsc19_soft.npy      # is the trellis structure right?
    python projects/atsc/atsc_diag.py edges     --soft data/atsc19_soft.npy      # are errors uniform, or at segment edges?
    python projects/atsc/atsc_diag.py rotation  --soft data/atsc19_soft.npy      # how do the 12 encoders advance per segment?
    python projects/atsc/atsc_diag.py levels    --soft data/atsc19_soft.npy      # is the gain/offset right, are the 8 levels even?
    python projects/atsc/atsc_diag.py layout    --soft data/atsc19_soft.npy      # which byte layout does Reed-Solomon accept?
    python projects/atsc/atsc_diag.py randomizer --soft data/atsc19_soft.npy     # does the randomizer keystream match the data?
    python projects/atsc/atsc_diag.py rs-errors --soft data/atsc19_soft.npy      # where in the stream are the corrected bytes?
    python projects/atsc/atsc_diag.py viterbi-selftest                           # does the Viterbi decoder work on synthetic data?

--soft is the cache a decode writes (atsc_decode.py --save-soft): soft symbols as <name>_soft.npy plus <name>_meta.npy.

The idea behind all of them: a decoder with many unverified assumptions fails silently ("0 good codewords"). Each check here
tests ONE assumption against the data and reports a number that is ~50% (chance) when it is wrong and well above when right.
What each one caught, in the order it was needed, is in projects/atsc/ATSC.md.
"""
import argparse
from collections import Counter
import itertools
import sys
import time

import numpy as np

import atsc_decode as ad
import atsc_probe as ap

SEG, FIELD = ap.SEG, ap.FIELD

def load(path):
    soft = np.load(path)
    meta = np.load(ad.meta_path(path))
    return soft, int(meta[0])

def data_segments(soft, fs):
    return np.array([k for k in range(soft.shape[0]) if k % FIELD != fs])

def level_bits(soft, ds):
    """Hard-slice the data of segments `ds` and split the level index k = 4*Z2 + 2*Z1 + Z0 into its bits (each (segments, 69, 12))"""
    k = ((ap.slice_levels(soft[ds, 4:]) + 7)/2).astype(int)
    r = lambda v: v.reshape(len(ds), 69, 12)
    return r((k >> 2) & 1), r((k >> 1) & 1), r(k & 1)

# ------------------------------------------------------------------------------------------------ checks

def relation(soft, fs, **_):
    """The trellis code's parity relation Z0[n] = Z0[n-2] xor Z1[n-1] on hard decisions, within segments.

    Holds ~90% when the Z bit mapping is right (errors of the slicer break it), exactly 50% for a wrong relation.
    Tests alternatives too, so a mismatch shows which structure is the real one. It does NOT test across segment
    boundaries (see `rotation`): in-segment structure can be right while the stream pairing is wrong."""
    ds = data_segments(soft, fs)
    z2, z1, z0 = level_bits(soft, ds)
    cands = {
        "z0[n] ^ z0[n-2] ^ z1[n-1]  (ATSC)": z0[:, 2:] ^ z0[:, :-2] ^ z1[:, 1:-1],
        "z0[n] ^ z0[n-2] ^ z1[n-2]": z0[:, 2:] ^ z0[:, :-2] ^ z1[:, :-2],
        "z0[n] ^ z0[n-1] ^ z1[n-1]": z0[:, 2:] ^ z0[:, 1:-1] ^ z1[:, 1:-1],
        "z0[n] ^ z0[n-2] ^ z1[n-1] ^ z2[n-1]": z0[:, 2:] ^ z0[:, :-2] ^ z1[:, 1:-1] ^ z2[:, 1:-1],
    }
    for name, e in cands.items():
        print(f"  {name:40s} holds {(1 - e.mean())*100:5.1f}%")
    print("  (polarity check: the same on inverted levels should give ~100 - that; a swap means the levels are upside down)")

def levels(soft, fs, **_):
    """Gain, offset and evenness of the eight levels. A fit slope != 1 means the soft values are mis-scaled (the slicer then
    mislabels the outer levels); uneven occupancy says the same. Calibrate with atsc_probe.fit_levels, never from sync symbols."""
    ds = data_segments(soft, fs)
    d = soft[ds, 4:].ravel()
    print(f"  mean {d.mean():+.3f}, std {d.std():.3f} (eight even levels: 4.58 + noise)")
    c = np.linspace(-7, 7, 8)
    for _ in range(30):
        a = np.argmin(np.abs(d[:, None] - c[None, :]), 1)
        c = np.array([d[a == i].mean() for i in range(8)])
    g, o = np.polyfit(np.linspace(-7, 7, 8), c, 1)
    print("  cluster centres: " + " ".join(f"{v:+.2f}" for v in c) + f"   -> centres = {g:.3f}*level {o:+.3f}  (want 1.000, 0)")
    print("  shares: " + " ".join(f"{np.mean(a == i)*100:.1f}" for i in range(8)) + "  (want 12.5 each)")
    sy = soft[ds, :4].mean(0)
    print(f"  mean sync symbols: {np.round(sy, 2)} (want +5 -5 -5 +5; an equalizer leaves ISI that biases these, so don't calibrate on them)")

def edges(soft, fs, **_):
    """Where in a segment does the Viterbi decoder disagree with plain slicing? Uniform ~2% is healthy (it is correcting
    the slicer's ~3% errors). Elevated at both ends and decaying geometrically (24, 14, 8, 4, 3 ... %) means the decoder is
    treating each segment as a separate trellis path: the encoder streams are paired wrongly across segment boundaries."""
    dib, ds, runs = _viterbi_continuous(soft, fs)
    ds = np.array(sorted(k for r in runs if len(r) == FIELD - 1 for k in r)[:6*(FIELD - 1)]) # The segments that were decoded
    z2, z1, z0 = level_bits(soft, ds)
    dis = (z1.reshape(len(ds), 828) != (dib[ds] & 1))
    print("  disagreement %, by step in segment (0..68), every 4th: " + " ".join(f"{v:.0f}" for v in dis.reshape(len(ds), 69, 12).mean((0, 2))[::4]*100))
    e = soft[ds, 4:] - ap.slice_levels(soft[ds, 4:])
    print(f"  soft error power by position, 6 bins: {np.round((e**2).reshape(len(ds), 6, 138).mean((0, 2)), 3)} (flat means the SYMBOLS are fine)")

def _viterbi_continuous(soft, fs):
    """The first-attempt decoder (encoders paired by position in the segment, ignoring the rotation): kept only so `edges`
    and `rotation` can show what the failure looks like"""
    nseg = soft.shape[0]
    ds = data_segments(soft, fs)
    runs, cur = [], [ds[0]]
    for k in ds[1:]:
        if k == cur[-1] + 1:
            cur.append(k)
        else:
            runs.append(cur); cur = [k]
    runs.append(cur)
    dib = np.zeros((nseg, 828), dtype=np.uint8)
    for r in [r for r in runs if len(r) == FIELD - 1][:6]:
        R = soft[r, 4:].reshape(len(r)*69, 12)[None]
        z1, x2 = ad.viterbi(R)
        dib[r] = (2*x2[0] + z1[0]).reshape(len(r), 828)
    return dib, ds, runs

def rotation(soft, fs, **_):
    """The trellis relation ACROSS a segment boundary, for each possible per-segment encoder rotation d (position e of segment s+1
    continues the stream at position e+d of segment s). One d gives ~90% and the rest 50%. For ATSC it is 4: the 12 encoders
    advance through all 832 symbol slots of a segment, the 4 uncoded sync symbols included, and 832 = 4 (mod 12)."""
    ds = data_segments(soft, fs)
    z2, z1, z0 = level_bits(soft, ds)
    cons = np.where(np.diff(ds) == 1)[0]
    for d in range(12):
        ok = np.mean([1 - (z0[cons + 1, 0, (e - d) % 12] ^ z0[cons, 67, e] ^ z1[cons, 68, e]).mean() for e in range(12)])
        print(f"  rotation {d:2d}: relation across the boundary holds {ok*100:5.1f}%" + ("   <--" if ok > 0.8 else ""))

def viterbi_selftest(**_):
    """Encode random data with the assumed trellis, add noise, decode: validates the Viterbi code itself, separately from whether
    the assumed trellis matches the real signal. Expect 0 errors at sigma 0.47 (MER 20 dB) and ~0.2% at sigma 0.8 (15 dB)."""
    rng = np.random.default_rng(1)
    F, T, E = 4, 6000, 12
    x1 = rng.integers(0, 2, (F, T, E)).astype(np.uint8); x2 = rng.integers(0, 2, (F, T, E)).astype(np.uint8)
    s1 = np.zeros((F, E), np.uint8); s2 = np.zeros((F, E), np.uint8); z2p = np.zeros((F, E), np.uint8)
    lev = np.zeros((F, T, E))
    for t in range(T):
        z2 = x2[:, t] ^ z2p
        lev[:, t] = 2*(4*z2.astype(int) + 2*x1[:, t].astype(int) + s1.astype(int)) - 7
        s1, s2 = (x1[:, t] ^ s2), s1
        z2p = z2
    for sigma in (0.47, 0.8):
        r = lev + rng.normal(0, sigma, lev.shape)
        z1, x2d = ad.viterbi(r.astype(np.float32))
        hard = np.clip(2*np.round((r - 1)/2) + 1, -7, 7)
        err = np.mean((z1[:, 5:] != x1[:, 5:]) | (x2d[:, 5:] != x2[:, 5:]))
        print(f"  sigma {sigma}: slicer symbol errors {np.mean(hard != lev)*100:.2f}%, Viterbi dibit errors {err*100:.3f}%")

# ------------------------------------------------------------------------------------------------ discovery

def layout(soft, fs, fields=5, **_):
    """Byte-layout hypotheses, each judged by Reed-Solomon on the SAME Viterbi output (the trellis paths follow the slot rotation of 4
    per segment; the question is how decoded 2-bit words become bytes). A valid RS codeword cannot happen by chance (~3e-7), so a
    hypothesis with correctable codewords is right, and the share says how right:
      A  bytes per encoder:           B = 12*j + encoder
      B  identity:                    4 consecutive symbols in time are a byte
      C  block layout by position:    in 12-segment blocks, 4 consecutive steps at one POSITION t%12 are byte 12*row + position   <- ATSC
      E  word-wise round robin over encoders (12 offsets)
    This is what found C: A and B and E gave 0 of 1508; C gave 1507 of 1508."""
    n = FIELD - 1
    nsym = n*828
    T = nsym//12
    s = np.arange(nsym)//828; t = np.arange(nsym) % 828
    order = np.argsort((4*s + 8 + t) % 12, kind="stable")
    R = np.stack([soft[fs + 1 + FIELD*f:fs + 1 + FIELD*f + n, 4:].reshape(-1)[order].reshape(12, T).T for f in range(fields)]).astype(np.float32)
    z1, x2 = ad.viterbi(R)
    dibL = (2*x2 + z1).astype(np.uint8) # per encoder stream
    dib_time = np.empty((fields, nsym), np.uint8)
    for f in range(fields):
        dib_time[f][order] = dibL[f].T.reshape(-1)
    pack = lambda d: (d[..., 0] << 6) | (d[..., 1] << 4) | (d[..., 2] << 2) | d[..., 3]
    def judge(name, per_field):
        z = ad.deinterleave(np.concatenate(per_field))
        ncw = (len(z) - ad.DELAY)//207
        _, err = ad.reed_solomon(z[:ad.DELAY + 207*ncw])
        print(f"  {name:62s} {int((err >= 0).sum()):5d} / {ncw} correctable", flush=True)
    judge("A  bytes per encoder, B = 12*j + L", [pack(dibL[f].T.reshape(12, T//4, 4)).T.reshape(-1) for f in range(fields)])
    judge("B  identity: 4 consecutive symbols in time", [pack(dib_time[f].reshape(-1, 4)) for f in range(fields)])
    def block_c(d):
        d = d.reshape(n//12, 207, 4, 12)
        return ((d[:, :, 0, :] << 6) | (d[:, :, 1, :] << 4) | (d[:, :, 2, :] << 2) | d[:, :, 3, :]).reshape(-1)
    judge("C  block layout by position: B = 12*row + t%12", [block_c(dib_time[f]) for f in range(fields)])
    for e0 in range(0, 12, 4):
        k = np.arange(T)[:, None]; L = np.arange(12)[None, :]
        W = (12*k + ((L - e0) % 12)).reshape(-1)
        out = []
        for f in range(fields):
            w = np.empty(T*12, np.uint8); w[W] = dibL[f].reshape(-1)
            out.append(pack(w.reshape(-1, 4)))
        judge(f"E  word-wise round robin over encoders, offset {e0}", out)

def randomizer(soft, fs, **_):
    """Test a randomizer (keystream) hypothesis against the data without knowing the plaintext. Needs the RS-corrected bytes
    from a decode. The first byte of a TS packet (after the removed 0x47) is the same value in almost every packet of a stream, so
    `keystream[i] xor observed[i]` must be one constant at every packet start if the keystream is right (~1/256 if wrong).
    Tests shifting the register once per byte (the figure's 'byte clock': right) against eight times (bit-serial: wrong)."""
    fields = [b for _, b in ad.decode_fields(soft, fs) if len(b) == ad.FIELD_DATA*207]
    data, err = ad.reed_solomon(ad.deinterleave(np.concatenate(fields)))
    nf = len(data)//ad.FIELD_DATA
    d = data[:nf*ad.FIELD_DATA].reshape(nf, ad.FIELD_DATA, 187)
    mode = np.array([Counter(d[:, k, 0].tolist()).most_common(1)[0][0] for k in range(ad.FIELD_DATA)], dtype=np.uint8)
    for shifts in (1, 8):
        ks = ad.randomizer_bytes(ad.FIELD_DATA*187, shifts)
        x = ks[::187][:ad.FIELD_DATA] ^ mode
        v, c = np.unique(x, return_counts=True)
        print(f"  {shifts} shift(s) per byte: keystream xor observed is constant for {c.max()/ad.FIELD_DATA*100:5.1f}% of packet starts (value {int(v[c.argmax()]):#04x})")

def rs_errors(soft, fs, c=0, **_):
    """Where do the corrected bytes sit? Maps each corrected byte of the RS-correctable codewords back to its place in the received
    stream. Errors in the first/last ~23 bytes of a segment => the edge problem (see `edges`); errors by deinterleaver branch
    n % 52 => the interleaver delays are wrong; errors by encoder (byte % 12) => one encoder is wrongly paired."""
    fields = [b for _, b in ad.decode_fields(soft, fs) if len(b) == ad.FIELD_DATA*207]
    y = np.concatenate(fields)
    z = ad.deinterleave(y)
    data, err = ad.reed_solomon(z)
    good = np.where(err >= 0)[0]
    print(f"  {len(err)} codewords, {len(good)} correctable ({len(good)/max(len(err), 1)*100:.1f}%)")
    cw = z[ad.DELAY:ad.DELAY + len(err)*207].reshape(-1, 207)
    diff = data[good] != cw[good, :187]
    ys = []
    for row, m in zip(diff, good):
        for j in np.where(row)[0]:
            n = ad.DELAY + 207*m + j
            ys.append(n - 208*(51 - n % 52))
    if not ys:
        print("  no corrections to map")
        return
    ys = np.array(ys)
    print("  by position in the received segment (bins of 23 bytes):", np.histogram(ys % 207, bins=np.arange(0, 230, 23))[0])
    print("  by encoder (byte % 12):", np.bincount(ys % 12, minlength=12))
    print("  by deinterleaver branch (byte % 52), 52 values:", np.bincount(ys % 52, minlength=52))

CHECKS = {"relation": relation, "levels": levels, "edges": edges, "rotation": rotation, "layout": layout, "rs-errors": rs_errors,
          "randomizer": randomizer, "viterbi-selftest": viterbi_selftest}

def main():
    ap_ = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap_.add_argument("check", choices=sorted(CHECKS))
    ap_.add_argument("--soft", metavar="NPY", help="Cached soft symbols (<name>_soft.npy, with <name>_meta.npy beside it)")
    ap_.add_argument("--c", type=int, default=0, help="Encoder offset for rs-errors")
    a = ap_.parse_args()
    if a.check != "viterbi-selftest" and not a.soft:
        ap_.error(f"{a.check} needs --soft")
    soft, fs = (None, None) if a.check == "viterbi-selftest" else load(a.soft)
    t0 = time.time()
    print(f"{a.check}:")
    CHECKS[a.check](soft=soft, fs=fs, c=a.c) if a.check != "viterbi-selftest" else viterbi_selftest()
    print(f"  ({time.time() - t0:.1f} s)")

if __name__ == "__main__":
    sys.exit(main())
