"""What decoded bits carry: known framings found at any bit offset, and a verdict on what is left

    found = rp.analysis.framing.payload.signatures(bits)     # [Found("E1 G.704", period 512, offset 37, ...), ...]
    v = rp.analysis.framing.payload.verdict(bits)            # "structured" / "scrambled" / "random-looking", and why

Each signature is a test whose chance rate is known, run at every bit offset, so a hit is judged against how often random
bits would give it:
  E1 (ITU G.704)      the frame alignment word x0011011 in time slot 0 of every other 256-bit frame (period 512)
  T1 (ITU G.704)      one framing bit every 193, repeating the superframe pattern (100011011100 for D4/SF, a 001011
                      pattern every 4th bit for ESF): a column of the 193-bit grid far from random
  ATM cells (I.432)   53-byte cells whose fifth byte is the CRC-8 (x^8 + x^2 + x + 1) of the first four, xor 0x55
  GFP (G.7041)        a core header whose 16-bit cHEC is the CRC-16 (x^16 + x^12 + x^5 + 1) of its length field, the four
                      bytes xored with 0xB6AB31E0, and the next header that length + 4 bytes on
  HDLC / PPP / LAPD   every run of six ones a flag 01111110 (bit stuffing), not half of them
  IPv4                20-byte headers starting 0x45 whose ones-complement checksum holds
  CCSDS               the attached sync marker 0x1ACFFC1D at a period
  MPEG-TS             0x47 every 188 bytes (204 with Reed-Solomon parity)
  fixed word          any bits constant at a period (frame structure in general, a unique word after demodulation,
                      bit errors and all), the period from the autocorrelation summed over its multiples

`verdict` combines them with the plain statistics: bias, byte distribution, autocorrelation, linear complexity over
windows (an additive LFSR scrambler shows as complexity much below half a window), and how well zlib compresses.
"""
import zlib

import numpy as np

from ...math.prob import deviance_z, z_critical

from ...math.gf2 import berlekamp_massey

class Found:
    """A signature found: name, period (bits, or None), offset (first bit), hits, expected (by chance), z, detail; a
    word found (`fixed_word`): word (hex, first bit sent most significant), bits, mismatches (median, 99%) per frame"""
    __slots__ = ("name", "period", "offset", "hits", "expected", "z", "detail", "word", "bits", "mismatches")
    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))

    def __str__(self):
        per = f", period {self.period} bits" if self.period else ""
        d = f"; {self.detail}" if self.detail else ""
        return f"{self.name}: offset {self.offset}{per}, {self.hits} hits vs {self.expected:.1f} by chance (z {self.z:.0f}){d}"

    __repr__ = __str__

def _z(hits, n, p):
    return (hits - n*p)/np.sqrt(max(n*p*(1 - p), 1e-12))

def _bytes_at(bits, off):
    b = np.asarray(bits[off:], dtype=np.uint8)
    return np.packbits(b[:len(b)//8*8])

def _crc(data, poly, width, init=0):
    """CRC (MSB first, no reflection) of each row of data (R, nbytes) uint8: (R,) ints"""
    data = np.asarray(data, dtype=np.int64)
    top = 1 << (width - 1)
    mask = (1 << width) - 1
    table = np.zeros(256, dtype=np.int64)
    for i in range(256):
        c = i << (width - 8)
        for _ in range(8):
            c = ((c << 1) ^ poly) & mask if c & top else (c << 1) & mask
        table[i] = c
    crc = np.full(data.shape[0], init, dtype=np.int64)
    for j in range(data.shape[1]):
        crc = ((crc << 8) & mask) ^ table[((crc >> (width - 8)) ^ data[:, j]) & 0xFF]
    return crc

def _window_bits(bits, length):
    """Every `length`-bit window of bits as an int (length <= 62)"""
    b = np.asarray(bits, dtype=np.int64)
    n = len(b) - length + 1
    if n <= 0:
        return np.zeros(0, np.int64)
    v = np.zeros(n, dtype=np.int64)
    for j in range(length):
        v = (v << 1) | b[j:j + n]
    return v

def _periodic_word(bits, word, wlen, periods, name, z_min):
    """`word` (wlen bits) at a fixed offset modulo a period"""
    v = _window_bits(bits, wlen)
    hit = v == word
    best = None
    for P in periods:
        if len(v) < 4*P:
            continue
        m = len(v)//P*P
        cols = hit[:m].reshape(-1, P).sum(axis=0)
        off = int(np.argmax(cols))
        h, n = int(cols[off]), m//P
        p = 2.0**-wlen
        # every offset was tried: the best column of P is compared with chance over n frames, by the binomial deviance (a
        # Gaussian z gives 2 hits where 0.06 are expected z 8) and past what the best of P random columns reaches
        z = float(deviance_z(h, n, p))
        if z > max(z_min, z_critical(P)) and (best is None or z > best.z):
            best = Found(name=name, period=P, offset=off, hits=h, expected=n*p, z=float(z),
                         detail=f"{100*h/n:.0f}% of {n} frames")
    return best

def e1(bits, z_min=8.0, share=0.6):
    """G.704 frame alignment: x0011011 every 512 bits (the first bit, Si, is free), in most frames (a framed stream's own
    fixed fields can hold the word at a period by chance, in some of them)"""
    f = _periodic_word(bits, 0b0011011, 7, (512,), "E1 G.704 (FAS)", z_min)
    if f is None or f.hits < share*(len(bits)//512):
        return None
    return f

def t1(bits, z_min=8.0):
    """A column of the 193-bit grid far from random: its bits repeat with the superframe (12 for SF, 4-bit steps in ESF)"""
    b = np.asarray(bits, dtype=np.uint8)
    P = 193
    n = len(b)//P
    if n < 48:
        return None
    G = b[:n*P].reshape(n, P).astype(np.int8)
    best = None
    for L in (12, 24):
        m = n//L*L
        pats = G[:m].reshape(-1, L, P) # (superframes, L, columns)
        agree = np.mean(pats == pats[:1], axis=(0, 1)) # per column: how often a superframe repeats the first's pattern
        # biased or framed data repeats patterns too: a column stands out against the other columns, not against 1/2
        med, mad = np.median(agree), np.median(np.abs(agree - np.median(agree))) + 1e-9
        col = int(np.argmax(agree))
        z = (agree[col] - med)/(1.4826*mad)
        if z > z_min and agree[col] > 0.9 and (best is None or z > best.z):
            best = Found(name=f"T1 G.704 ({'SF' if L == 12 else 'ESF'} framing bit)", period=P, offset=col,
                         hits=int(agree[col]*m), expected=float(med*m), z=float(z),
                         detail=f"pattern {''.join(map(str, pats[0, :, col]))}")
    return best

def atm(bits, z_min=8.0):
    """53-byte cells: HEC = CRC-8(x^8+x^2+x+1) of the first four header bytes, xor 0x55 (I.432), at some bit offset"""
    best = None
    for off in range(8):
        B = _bytes_at(bits, off)
        if len(B) < 53*20:
            continue
        n = len(B) - 4
        hdr = np.lib.stride_tricks.sliding_window_view(B, 5)[:n]
        ok = (_crc(hdr[:, :4], 0x07, 8) ^ 0x55) == hdr[:, 4]
        m = len(ok)//53*53
        cols = ok[:m].reshape(-1, 53).sum(axis=0)
        k = int(np.argmax(cols))
        z = _z(int(cols[k]), m//53, 1/256)
        if z > z_min and (best is None or z > best.z):
            best = Found(name="ATM cells (HEC)", period=424, offset=8*k + off, hits=int(cols[k]), expected=m/53/256,
                         z=float(z), detail=f"{100*cols[k]/(m//53):.0f}% of cells")
    return best

def gfp(bits, z_min=8.0):
    """GFP core headers: PLI (16) + cHEC = CRC-16 of PLI, the four bytes xored with 0xB6AB31E0; frames chained by PLI"""
    best = None
    for off in range(8):
        B = _bytes_at(bits, off).astype(np.int64)
        if len(B) < 256:
            continue
        w = np.lib.stride_tricks.sliding_window_view(B, 4)
        hdr = w ^ np.array([0xB6, 0xAB, 0x31, 0xE0])[None, :]
        ok = _crc(hdr[:, :2], 0x1021, 16) == ((hdr[:, 2] << 8) | hdr[:, 3])
        # chained: a header PLI + 4 bytes after another
        idx = np.flatnonzero(ok)
        pli = (hdr[idx, 0] << 8) | hdr[idx, 1]
        nxt = idx + pli + 4
        chain = np.isin(nxt, idx).sum()
        z = _z(int(chain), len(idx) + 1, 2.0**-16*max(1, len(idx)))
        if chain >= 4 and (best is None or chain > best.hits):
            best = Found(name="GFP (cHEC, chained)", period=None, offset=8*int(idx[0]) + off, hits=int(chain),
                         expected=len(w)*2.0**-16*len(idx)*2.0**-16, z=float(min(z, 1e6)), detail=f"{len(idx)} headers")
    return best if best is not None and best.hits >= 4 else None

def hdlc(bits, z_min=8.0):
    """HDLC / PPP / LAPD: bit stuffing leaves no run of six ones but a flag's (01111110), where in random bits half of the
    windows 0111111 go on with a 1"""
    v8 = _window_bits(bits, 8)
    flags = int(np.sum(v8 == 0x7E))
    six = flags + int(np.sum(v8 == 0x7F)) # 0111111 then 0 (a flag) or 1
    if flags < 8:
        return None
    z = _z(flags, six, 0.5)
    if z > z_min:
        return Found(name="HDLC flags (bit-stuffed)", period=None, offset=int(np.argmax(v8 == 0x7E)), hits=flags,
                     expected=six/2, z=float(z), detail=f"{100*flags/max(six, 1):.0f}% of 0111111 runs end in a flag")
    return None

def ipv4(bits, z_min=8.0):
    """20-byte IPv4 headers: 0x45, total length >= 20, ones-complement checksum of the ten words 0xFFFF"""
    best = None
    for off in range(8):
        B = _bytes_at(bits, off).astype(np.int64)
        if len(B) < 40:
            continue
        cand = np.flatnonzero(B[:-20] == 0x45)
        if not len(cand):
            continue
        H = B[cand[:, None] + np.arange(20)[None, :]]
        w = (H[:, 0::2] << 8) | H[:, 1::2]
        s = w.sum(axis=1)
        while np.any(s >> 16):
            s = (s & 0xFFFF) + (s >> 16)
        tl = w[:, 1]
        ok = (s == 0xFFFF) & (tl >= 20)
        h = int(ok.sum())
        z = _z(h, len(cand), 2.0**-16)
        if h >= 3 and z > z_min and (best is None or z > best.z):
            best = Found(name="IPv4 headers", period=None, offset=8*int(cand[ok][0]) + off, hits=h,
                         expected=len(cand)*2.0**-16, z=float(z), detail=f"{h} with valid checksums")
    return best

def ccsds(bits, z_min=8.0, periods=None):
    """The attached sync marker 0x1ACFFC1D at a period (any of `periods`, default every multiple of 8 up to 16384 found
    from its own hits)"""
    v = _window_bits(bits, 32)
    idx = np.flatnonzero(v == 0x1ACFFC1D)
    if len(idx) < 3:
        return None
    d = np.diff(idx)
    P = int(np.bincount(d).argmax())
    h = int(np.sum(d == P))
    return Found(name="CCSDS ASM 0x1ACFFC1D", period=P, offset=int(idx[0]), hits=len(idx), expected=len(v)*2.0**-32,
                 z=float(_z(len(idx), len(v), 2.0**-32)), detail=f"{h} gaps of {P} bits")

def mpeg_ts(bits, z_min=8.0):
    """0x47 every 188 or 204 bytes"""
    best = None
    for P in (188*8, 204*8):
        f = _periodic_word(bits, 0x47, 8, (P,), f"MPEG-TS ({P//8}-byte packets)", z_min)
        if f is not None and (best is None or f.z > best.z):
            best = f
    return best

WORD_AGREE = 0.9 # fixed_word: how often a word's position reaches `agree` (bit errors take some short of it)
WORD_DENSITY = 0.75 # and the least share of its positions that do: under it, structure of another kind (a scrambler's
# keystream folded at a divisor of its period has columns agreeing far beyond chance, scattered, half of them)

def fixed_word(bits, max_period=16384, z_min=8.0, min_bits=8, agree=0.75, alpha=1e-3, false_alarm=1e-6):
    """Bits constant at a period, as a unique word after demodulation: Found with the period, the word (its longest run
    of constant positions: `word`, a hex string sent first bit first, `bits` long, at `offset`), `hits` (every constant
    position), and in `mismatches` the bits each frame's copy differs by (median, 99th percentile)

    The candidate periods sum the bits' autocorrelation over the multiples of each lag (frames.comb_lags), so a short
    word shows within a few dozen frames. A position is constant when at least `agree` of the frames hold the same bit
    there, which channel bit errors mostly leave it (an exact agreement they would not); one such position is common by
    chance over few frames, a stretch of them is not. The word is the stretch where they gather most (one position bit
    errors left short breaks it not), taken when its chance over every position and candidate looked at is under
    `alpha`. The whole stream inverted keeps its words (the
    word given is as received). `errors` in `detail` is what a delimit step could allow: a margin over the 99th
    percentile, no more than rpps.sync.word_errors at `false_alarm` over a period of positions."""
    from math import log, lgamma
    from ...math.prob import log_binom_half_cdf
    from ...sync.frame import word_errors
    from .frames import comb_lags, HALF_NORMAL_MEAN, HALF_NORMAL_VAR
    b = np.asarray(bits, dtype=np.uint8)
    n = len(b)
    if n < 4*64:
        return None
    x = 1 - 2*b.astype(np.float64)
    x = x - x.mean()
    v = max(float(np.mean(x**2)), 1e-12)
    F = np.fft.rfft(x, 1 << int(np.ceil(np.log2(2*n))))
    za = np.abs(np.fft.irfft(np.abs(F)**2)[:n])/(v*np.sqrt(n - np.arange(n)))
    cands = [L for L, _ in comb_lags(za, 8, max_period, mean=HALF_NORMAL_MEAN, var=HALF_NORMAL_VAR)]
    cands = sorted({c//d for c in cands for d in range(1, 7) if c % d == 0 and c//d >= 8}) # their submultiples too
    best = None
    for P in cands:
        m = n//P
        if m < 8:
            continue
        G = b[:m*P].reshape(m, P)
        ones = G.sum(axis=0, dtype=np.int64)
        k = int(np.ceil(agree*m)) # (frames that must agree)
        lp = min(0.0, log(2) + log_binom_half_cdf(m, m - k)) # log chance a random position agrees so, either way
        hot = np.maximum(ones, m - ones) >= k
        if hot.sum() < min_bits or lp >= 0:
            continue
        if lp >= log(WORD_AGREE):
            continue
        # The word: the segment (circular: it may straddle the fold) scoring most when a constant position counts for
        # it and any other against, as likely in a word as WORD_AGREE (a position its bit errors leave short of `agree`
        # breaks no word); then taken only if chance could not have made it, over every position and candidate
        h2 = np.concatenate((hot, hot))
        sc = np.where(h2, log(WORD_AGREE) - lp, log(1 - WORD_AGREE) - np.log1p(-np.exp(lp)))
        S = np.concatenate(([0.0], np.cumsum(sc)))
        lo = np.minimum.accumulate(S)
        at = np.arange(len(S))
        lo_at = np.maximum.accumulate(np.where(S <= lo, at, 0)) # (where the running minimum was last reached)
        gain = S - lo
        e = int(np.argmax(gain))
        s0 = int(lo_at[e])
        ln = min(e - s0, P)
        hits = int(h2[s0:s0 + ln].sum())
        chance = lgamma(ln + 1) - lgamma(hits + 1) - lgamma(ln - hits + 1) + hits*lp # (ln positions, hits of them constant)
        if hits < min_bits or hits < WORD_DENSITY*ln or chance > log(alpha) - log(P*len(cands)):
            continue
        s0 %= P
        cols = (s0 + np.arange(ln)) % P
        colz = np.abs(2*ones - m)/np.sqrt(m)
        z = float(min(np.sum(colz[cols])/np.sqrt(ln), 1e6)) # (Stouffer, over the word's positions)
        if z < z_min:
            continue
        word = 2*ones[cols] > m
        rows = G[:, cols] if s0 + ln <= P else b[s0:s0 + (m - 1)*P].reshape(m - 1, P)[:, :ln] # (frames whose copy is whole)
        mis = np.count_nonzero(rows != word[None, :], axis=1)
        med, p99 = float(np.median(mis)), float(np.percentile(mis, 99))
        allowed = word_errors(ln, false_alarm, P)
        errors = min(max(int(np.ceil(2*p99)), int(p99) + 2), allowed) if allowed >= 0 else -1
        value = int("".join("1" if w else "0" for w in word), 2)
        f = Found(name="fixed word at a period", period=P, offset=int(s0), hits=int(hits),
                     expected=float(P*len(cands)*np.exp(chance)), z=z, word=f"{value:#0{(ln + 3)//4 + 2}x}", bits=int(ln),
                     mismatches=(med, p99),
                     detail=f"{int(hits)} constant bit positions; a {ln}-bit word at {s0}, each frame's copy off by "
                            f"{med:.0f} bits (99%: {p99:.0f}); errors {errors} for a delimit step"
                            + ("" if allowed >= 0 else f" (too short to search {P} positions at {false_alarm:g})"))
        if best is None or chance < best[0]: # The word chance is least likely to make: a multiple of the period, over
            best = (chance, f)              # fewer frames, or a divisor, where other frames' data dilutes it, makes it less
    return None if best is None else best[1]

TESTS = (e1, t1, atm, gfp, hdlc, ipv4, ccsds, mpeg_ts, fixed_word)

def signatures(bits, z_min=8.0, tests=TESTS, max_bits=2_000_000):
    """Every signature that holds on the bits (see the module), strongest first (on the first `max_bits`)"""
    bits = np.asarray(bits, dtype=np.uint8)[:max_bits]
    out = [f for t in tests if (f := t(bits, z_min=z_min)) is not None]
    return sorted(out, key=lambda f: -f.z)

class Verdict:
    """Result of `verdict`: kind ("structured", "scrambled", "random-looking"), why, signatures, bias (|mean - 1/2|*2),
    byte_chi (z of the byte distribution, the worst of 8 offsets), lc (linear complexity over windows / their length),
    compress (zlib size / size)"""
    __slots__ = ("kind", "why", "signatures", "bias", "byte_chi", "lc", "compress", "repeats", "randomness")
    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))

    def __str__(self):
        rep = f"; {self.repeats} repeated frames set aside" if self.repeats else ""
        return (f"{self.kind}: {self.why} (bias {self.bias:.3f}, bytes z {self.byte_chi:.1f}, linear complexity "
                f"{self.lc:.2f} of a window, zlib {self.compress:.2f}{rep})")

    __repr__ = __str__

def verdict(bits, window=512, z_min=8.0, frame=None, max_bits=2_000_000):
    """structured (a signature, bias, uneven bytes, or zlib compresses it), scrambled (an LFSR: windows' linear
    complexity well under half their length), or random-looking (encrypted, compressed, or under something not linear).
    frame: bits a frame (or frames (F, n) given): whole frames that repeat (idle fill) are set aside first, as they would
    make random data compress. On the first `max_bits`"""
    b = np.asarray(bits, dtype=np.uint8)
    repeats = 0
    if b.ndim == 2 or frame:
        F = b if b.ndim == 2 else b[:len(b)//frame*frame].reshape(-1, frame)
        _, first = np.unique(np.packbits(F, axis=1), axis=0, return_index=True)
        repeats = len(F) - len(first)
        b = F[np.sort(first)].reshape(-1)
    b = b[:max_bits]
    sig = signatures(b, z_min)
    bias = float(abs(2*b.mean() - 1))
    chi = 0.0
    for off in range(8):
        B = _bytes_at(b, off)
        if len(B) < 512:
            continue
        c = np.bincount(B, minlength=256)
        e = len(B)/256
        chi = max(chi, (float(((c - e)**2).sum()/e) - 255)/np.sqrt(510))
    lcs = [berlekamp_massey(b[a:a + window])[1]/window for a in range(0, len(b) - window + 1, window)][:16]
    lc = float(np.median(lcs)) if lcs else 0.5
    raw = np.packbits(b[:len(b)//8*8]).tobytes()
    comp = len(zlib.compress(raw, 9))/max(len(raw), 1)
    if sig:
        kind, why = "structured", "; ".join(str(s) for s in sig[:3])
    elif bias*np.sqrt(len(b)) > z_min or chi > z_min or comp < 0.95:
        kind, why = "structured", "uneven bits or bytes, or it compresses"
    elif lc < 0.4:
        kind, why = "scrambled", f"an LFSR of about {int(lc*window)} stages makes {window}-bit windows of it"
    else:
        kind, why = "random-looking", "no framing, even bits and bytes, full linear complexity, incompressible: " \
                                      "encrypted, compressed, or under a layer not found"
    rnd = None
    if kind == "random-looking": # The NIST SP 800-22 tests tell a cipher's output from most compressors' (randomness)
        from .randomness import test as _randomness
        rnd = _randomness(b)
        if rnd is not None and rnd.passed:
            why = (f"no framing, full linear complexity, and it passes the NIST SP 800-22 tests ({rnd.entropy:.2f} bits a "
                   "byte): as random as a cipher's output: encrypted or keyed (or a strong compressor's output)")
        elif rnd is not None:
            why = (f"no framing, full linear complexity, {rnd.entropy:.2f} bits a byte, but it fails "
                   f"{', '.join(rnd.failed)}: compressed, or a weak cipher or scrambler")
    return Verdict(kind=kind, why=why, signatures=sig, bias=bias, byte_chi=chi, lc=lc, compress=comp, repeats=repeats,
                   randomness=rnd)
