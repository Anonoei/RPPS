"""Decode an ATSC (8VSB) channel from a capture to an MPEG transport stream

    python projects/atsc/atsc_decode.py data/atsc19.cf64 --fs 20 --cf 507 --lower 500 --out ch19.ts

atsc_probe.recover() gives equalized soft symbols; this is the rest of the receiver (ATSC A/53 part 2):
  1. trellis: 12 interleaved 4-state encoders, 2 bits per symbol. Viterbi decoding on the soft levels.
  2. the bits of each segment are 207 bytes (187 data + 20 Reed-Solomon parity), packed MSB first
  3. de-interleave: a convolutional byte interleaver, 52 branches of 4 bytes, restarted at every field
  4. Reed-Solomon (207,187), t=10, corrects up to 10 wrong bytes per segment (rp.code.RS)
  5. derandomize: XOR with a 16-bit PRBS restarted at every field; the 0x47 sync byte goes back on the front
"""
import argparse
import os
import sys
import time
from collections import Counter
from types import SimpleNamespace

import numpy as np

import rpps as rp
import atsc_probe as ap

SEG, FIELD = ap.SEG, ap.FIELD
LEVELS = 2*np.arange(8) - 7.0 # Trellis output k = 4*Z2 + 2*Z1 + Z0 is level 2k - 7
BRANCHES, BRANCH_MULT = 52, 4 # Convolutional byte interleaver (B, M)
DELAY = BRANCHES*BRANCH_MULT*(BRANCHES - 1) # 10608 bytes: every byte is delayed by this much through interleave + deinterleave

# ------------------------------------------------------------------------------------------------ trellis

_TCM = rp.code.load("block", "trellis/atsc_8vsb")._dec # 8 states: the precoder's last Z2 and the 4-state code
_RS = rp.code.RS(207, 187)._dec # t = 10; rpps._native decodes the codewords in C++ on several threads (RPPS_NO_NATIVE=1: Python loops)


def rs_decode(cw):
    """Reed-Solomon (207, 187) over many codewords at once: cw (N, 207) uint8 -> (data (N, 187) uint8, errors (N,) int: corrected bytes, -1 = uncorrectable, data as received)"""
    cw = np.ascontiguousarray(cw, dtype=np.int64)
    data = np.asarray(_RS.decode(cw.reshape(-1), symbols=True)).reshape(-1, 187).astype(np.uint8)
    return data, np.asarray(_RS.errors, dtype=int)


def viterbi(R, raw=False):
    """Soft decisions for the trellis code (rpps block/trellis/atsc_8vsb). R: (runs, steps, 12) soft levels, 12 interleaved encoders, one symbol per step each.
    Returns (z1, x2): the Z1 (= X1) bit and X2 bit of each symbol, as (runs, steps, 12) uint8; raw=True gives (z1, z2), Z2 before the precoder is undone.

    The code (ATSC A/53): Z1 = X1, Z0 = s1, with the state updating s1' = X1 xor s2, s2' = s1; Z2 = X2 xor the last Z2. A level is
    k = 4*Z2 + 2*Z1 + Z0 (amplitude 2k - 7). Each encoder's run is decoded on its amplitudes from any start state to the best end state.
    """
    F, T, E = R.shape
    flat = np.asarray(R, np.float64).transpose(0, 2, 1).reshape(F*E, T)
    bits = _TCM.decode_batch(flat) # (F*E, 2T): X2, X1 per step (rpps._native when built)
    to = lambda v: v.astype(np.uint8).reshape(F, E, T).transpose(0, 2, 1)
    z1 = to(bits[:, 1::2])
    if raw:
        return z1, to((_TCM._path_out >> 2) & 1)
    return z1, to(bits[:, 0::2])

def meta_path(soft_path):
    """Where the metadata of a soft-symbol cache lives: <name>_soft.npy -> <name>_meta.npy, otherwise <name>_meta.npy beside any other name (a name without "_soft"
    used to map to itself, so the soft symbols were read back as the metadata)"""
    base = soft_path[:-4] if soft_path.endswith(".npy") else soft_path
    base = base[:-5] if base.endswith("_soft") else base
    return base + "_meta.npy"

def decode_fields(soft, fs_seg):
    """Trellis-decode every field in `soft` (segments x 832 levels) to bytes. Returns a list of (first segment, bytes) per field.

    Two different things have to be right, and they are not the same rotation:
    * The trellis PATHS. Which of the 12 encoders makes a symbol is decided by the slot it is sent in, and the slots count all 832
      symbols of a segment, the 4 uncoded sync symbols included (832 = 4 mod 12). Slot S of the field (S = 0 at the first symbol of
      the field sync segment) is encoder S % 12, so data symbol t of data segment s (s = 0 for the first after the field sync)
      is encoder (4*s + 8 + t) % 12. Each encoder's symbols, in time order, are one trellis path, decoded by `viterbi`.
    * The BYTE layout, which follows the position in the segment instead: in a block of 12 segments, step m = 69*s + t//12
      at position e = t % 12 is a 2-bit word, and four consecutive steps at one position (MSB pair first) are byte 12*(m//4) + e
      of the block (2484 bytes = 12 segments of 207). Found by trying the alternatives against Reed-Solomon (atsc_diag.py layout)."""
    nseg = soft.shape[0]
    jobs = []
    for k0 in [k + 1 for k in range(nseg) if k % FIELD == fs_seg]: # First data segment of each field
        n = min(k0 + FIELD - 1, nseg) - k0 # Segments k0 .. k0+n-1 are data (the next field sync is at k0 + 312)
        if n >= 24:
            jobs.append((k0, n))
    out = {}
    def go(group):
        n = group[0][1]
        nsym = n*828
        T = nsym//12
        s = np.arange(nsym)//828; t = np.arange(nsym) % 828
        order = np.argsort((4*s + 8 + t) % 12, kind="stable") # Each encoder's symbols, in time order
        R = np.stack([soft[k0:k0 + n, 4:].reshape(-1)[order].reshape(12, T).T for k0, _ in group]).astype(np.float32)
        z1, x2 = viterbi(R)
        for i, (k0, _) in enumerate(group):
            per_encoder = (2*x2[i] + z1[i]).astype(np.uint8).T.reshape(-1) # (12, T) flattened: encoder streams in label order
            dib = np.empty(nsym, dtype=np.uint8)
            dib[order] = per_encoder # Back to the order the symbols were sent
            nb = n//12*12 # Whole blocks of 12 segments
            d = dib[:nb*828].reshape(nb//12, 12*69, 12).reshape(nb//12, 207, 4, 12) # (block, byte row, word, position)
            out[k0] = ((d[:, :, 0, :] << 6) | (d[:, :, 1, :] << 4) | (d[:, :, 2, :] << 2) | d[:, :, 3, :]).reshape(-1)
    full = [j for j in jobs if j[1] == FIELD - 1]
    if full:
        go(full)
    for j in jobs:
        if j[1] != FIELD - 1:
            go([j])
    return [(k0, out[k0]) for k0, _ in jobs]

def deinterleave(y):
    """Convolutional byte deinterleaver: branch j (of 52) has a delay of 4*(51-j) of its own bytes, i.e. 208*(51-j) in the stream.
    z[n] = y[n - 208*(51 - n % 52)], valid from n = 10608 (the stream must start at a field, which restarts the commutator)"""
    n = np.arange(len(y))
    idx = n - BRANCHES*BRANCH_MULT*(BRANCHES - 1 - n % BRANCHES)
    z = np.zeros_like(y)
    ok = idx >= 0
    z[ok] = y[idx[ok]]
    return z

def reed_solomon(z, start=DELAY):
    """RS(207,187) on the deinterleaved stream. Returns (data (codewords, 187), errors per codeword (-1: uncorrectable))"""
    code = rp.code.RS(207, 187) # ATSC: GF(256) with 0x11D, roots alpha^0 .. alpha^19: the defaults
    rs = code._dec # The byte-level code inside the pipeline stage: takes bytes (symbols=True), reports `errors` per codeword
    cw = z[start:start + (len(z) - start)//207*207]
    out = rs.decode(cw.astype(np.int64), symbols=True)
    return np.asarray(out).reshape(-1, 187).astype(np.uint8), np.asarray(rs.errors)

# ------------------------------------------------------------------------------------------------ derandomizer, transport stream

RANDOMIZER_ADDERS = frozenset({1, 3, 6, 7, 11, 12, 13}) # Figure 5.3: G(16) = x16 + x13 + x12 + x11 + x7 + x6 + x3 + x + 1, feedback added after these stages
RANDOMIZER_TAPS = (1, 3, 4, 7, 11, 12, 13, 14) # D0..D7 are the outputs of these stages, read in parallel (D7 XORs the data MSB)

def randomizer_bytes(n, shifts_per_byte=1):
    """The first n bytes of the ATSC randomizer (A/53 part 2, figure 5.3). A 16-stage Galois shift register preloaded to 0xF180
    (stages 16, 15, 14, 13, 9, 8 set); one byte is read from the eight tapped stages and the register is shifted ONCE per byte
    (the figure says 'shifted with the Byte Clock'; shifting eight times, as a bit-serial LFSR would, is wrong: tried, ~chance).
    It runs over the data bytes only (the segment sync / MPEG sync byte is not randomized), restarting at each field."""
    s = [0]*17
    for k in (16, 15, 14, 13, 9, 8):
        s[k] = 1
    out = np.empty(n, dtype=np.uint8)
    for i in range(n):
        out[i] = sum(s[t] << b for b, t in enumerate(RANDOMIZER_TAPS))
        for _ in range(shifts_per_byte):
            fb = s[16]
            s = [0, fb] + [s[k] ^ (fb if k in RANDOMIZER_ADDERS else 0) for k in range(1, 16)]
    return out

def derandomize(data):
    """data: (codewords, 187) RS-corrected bytes, codeword m being segment m % 312 of its field. Returns the same shape, plain."""
    key = randomizer_bytes(FIELD_DATA*187).reshape(FIELD_DATA, 187)
    return data ^ key[np.arange(len(data)) % FIELD_DATA]

FIELD_DATA = FIELD - 1 # Data segments per field

def transport_stream(plain, errors):
    """Put the 0x47 sync byte back. An uncorrectable codeword gets the transport-error bit set, as a receiver would."""
    ts = np.empty((len(plain), 188), dtype=np.uint8)
    ts[:, 0] = 0x47
    ts[:, 1:] = plain
    bad = errors < 0
    ts[bad, 1] |= 0x80
    return ts

def ts_report(ts):
    """What is in a transport stream: PIDs, and the continuity-counter check that proves the packets are in order and intact"""
    pid = ((ts[:, 1].astype(int) & 0x1F) << 8) | ts[:, 2]
    cc = ts[:, 3] & 0x0F
    afc = (ts[:, 3] >> 4) & 3 # 1 payload, 2 adaptation only, 3 both
    lines = []
    ok = tot = 0
    last = {}
    for i in range(len(ts)):
        p = int(pid[i])
        if p == 0x1FFF:
            continue
        if p in last and afc[i] in (1, 3):
            tot += 1
            ok += int(cc[i]) == (last[p] + 1) % 16
        if afc[i] in (1, 3):
            last[p] = int(cc[i])
    c = Counter(pid.tolist())
    return c, ok, tot

def ts_from_soft(soft, fs_seg):
    """Soft symbols -> transport stream packets (trellis, de-interleave, Reed-Solomon, derandomize), printing each stage's statistics"""
    print(f"\n{soft.shape[0]} segments, field sync at {fs_seg} of {FIELD}")
    t = time.time()
    fields = [b for _, b in decode_fields(soft, fs_seg) if len(b) == FIELD_DATA*207] # Whole fields: the interleaver restarts at each
    print(f"Trellis (Viterbi) decoded {len(fields)} whole fields in {time.time() - t:.1f} s")
    z = deinterleave(np.concatenate(fields))
    data, err = reed_solomon(z)
    good = err >= 0
    print(f"Reed-Solomon: {len(err)} segments, {good.sum()} correctable ({good.mean()*100:.2f}%), mean {err[good].mean():.1f} bytes corrected each "
          f"(limit 10; {np.sum(err == 10)} at the limit)")
    ts = transport_stream(derandomize(data), err)
    c, ok, tot = ts_report(ts)
    print(f"Transport stream: {len(ts)} packets; continuity counters in order for {ok}/{tot} ({ok/max(tot, 1)*100:.1f}%)")
    print("PIDs (packets): " + ", ".join(f"0x{p:04X}: {n}" for p, n in c.most_common(12)))
    return ts


def main():
    ap_ = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap_.add_argument("path")
    ap_.add_argument("--fs", type=float, required=True, help="Sample rate [MS/s]")
    ap_.add_argument("--cf", type=float, required=True, help="Capture centre [MHz]")
    ap_.add_argument("--lower", type=float, required=True, help="Channel's lower edge [MHz]")
    ap_.add_argument("--out", metavar="TS", help="Write the transport stream here")
    ap_.add_argument("--chunk", type=float, metavar="SECONDS", help="Decode the capture in pieces of this length and join the streams. No longer needed for quality (the equalizer now tracks the channel in blocks of "
                     "atsc_probe.BLOCK_FIELDS fields); only to bound memory on very long captures")
    ap_.add_argument("--soft", metavar="NPY", help="Use these soft symbols (cached from a previous run) instead of demodulating")
    ap_.add_argument("--save-soft", metavar="NPY", help="Save the soft symbols (and <name>_meta.npy) for atsc_diag.py and a quick re-run")
    a = ap_.parse_args()
    if a.chunk and not a.soft:
        n = int(a.chunk*a.fs*1e6); total = os.path.getsize(a.path)//8; tss = []
        for k in range(0, total - n//4, n):
            x = np.fromfile(a.path, np.complex64, count=n, offset=k*8)
            r = ap.recover(x, a.fs*1e6, a.cf*1e6, a.lower*1e6)
            if r.fs_seg is None:
                print(f"\n--- piece at {k/(a.fs*1e6):.1f} s: no field sync, skipped"); continue
            print(f"\n--- piece at {k/(a.fs*1e6):.1f} s")
            tss.append(ts_from_soft(r.soft, r.fs_seg))
        if not tss:
            raise SystemExit("No field sync found in any piece: not a decodable ATSC capture")
        ts = np.concatenate(tss)
    else:
        if a.soft:
            soft = np.load(a.soft); meta = np.load(meta_path(a.soft)); fs_seg = int(meta[0])
        else:
            x = np.fromfile(a.path, np.complex64)
            r = ap.recover(x, a.fs*1e6, a.cf*1e6, a.lower*1e6)
            if r.fs_seg is None:
                raise SystemExit("No field sync found: not a decodable ATSC capture")
            soft, fs_seg = r.soft, r.fs_seg
            if a.save_soft:
                np.save(a.save_soft, soft); np.save(meta_path(a.save_soft), np.array([fs_seg, r.nseg, r.start, r.first]))
        ts = ts_from_soft(soft, fs_seg)
    if a.out:
        ts.tofile(a.out)
        print(f"Wrote {a.out} ({ts.size/1e6:.2f} MB)")
    return ts

if __name__ == "__main__":
    main()
