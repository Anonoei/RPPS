"""P25 phase 1 voice to audio, for calls that are not encrypted: LDU frames of one channel -> IMBE decoder (imbe.py) -> a WAV file

    python projects/p25_dmr/p25_audio.py data/lmr/p25_854.cf64 --fs 8 --cf 854 --freq 853.575 --wav data/voice3/853.5625.wav

Per channel: find the P25 frame syncs (p25_sync), take the LDU1 / LDU2 frames, read each LDU2's encryption algorithm (p25_voice: 0x80 = unencrypted) and decode the nine 72-symbol voice frames of every LDU
with the IMBE decoder. A call (a run of LDUs with gaps under 0.3 s) is decoded only when its LDU2s mostly say unencrypted (at least two, a majority, and no other algorithm twice: one frame can be misread); anything else is reported and skipped (an encrypted frame decodes
to noise, and no key is used or attempted). Voice frames sit at the symbols (without status symbols) 56, 128, 220, 312, 404, 496, 588, 680 and 768 of the 840-symbol LDU; their bit layout is Annex H of TIA-102.BABA
(imbe.py); the positions were checked against an independent implementation's table. Output: 8 kS/s mono 16-bit WAV, silence between calls, written locally, never uploaded. Meant to be upstreamed into RPPS.
"""
import argparse
import wave
from collections import Counter

import numpy as np

import imbe
import p25_sync
import p25_voice

STARTS = [56, 128, 220, 312, 404, 496, 588, 680, 768] # first symbol of each voice frame in the LDU, status symbols removed


def voice_frames(fr):
    """The nine 144-bit voice frames of an LDU from its 864 level-calibrated symbols (list of 9 lists of bits)"""
    keep = [i for i in range(864) if (i + 1) % 36 != 0]
    d = p25_sync.slice_dibits(np.asarray(fr)[keep])
    out = []
    for s in STARTS:
        bits = []
        for v in d[s:s + 72]:
            bits += [(int(v) >> 1) & 1, int(v) & 1]
        out.append(bits)
    return out


def calls(x, fs, cf, freq, gap=0.3):
    """Group the LDUs of a channel into calls: list of dicts(start s, ldus [(time, duid, frames)], algs Counter)"""
    sn, c = p25_sync.symbols(x, fs, cf, freq); stream = p25_sync.symbols.stream
    hits, _ = p25_sync.sync_hits(sn)
    cl = []; last = None
    for i in hits:
        fr = p25_sync.refine(stream, i, n=864)
        if len(fr) < 864:
            continue
        v = p25_sync.nid(fr, 0)
        if not v or v[1] not in (0x5, 0xA):
            continue
        g = np.mean(fr[:24]*p25_sync.SYNC)/9.0
        fr = fr/g; t = i/4800.0
        if last is None or t - last > gap:
            cl.append(dict(start=t, ldus=[], algs=Counter()))
        last = t
        if v[1] == 0xA:
            _, alg, _ = p25_voice.ess(p25_voice.hexbits(p25_voice.frame_bits_full(fr)))
            cl[-1]["algs"][alg] += 1
        cl[-1]["ldus"].append((t, v[1], voice_frames(fr)))
    return cl


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file"); ap.add_argument("--fs", type=float, required=True, help="[MS/s]"); ap.add_argument("--cf", type=float, required=True, help="[MHz]")
    ap.add_argument("--freq", type=float, required=True); ap.add_argument("--seconds", type=float, default=20.0); ap.add_argument("--wav", help="write the decoded audio here")
    a = ap.parse_args(argv)
    fs = a.fs*1e6; x = np.fromfile(a.file, np.complex64, count=int(fs*a.seconds))
    cl = calls(x, fs, a.cf*1e6, a.freq*1e6)
    if not cl:
        print("no voice frames (LDU1 / LDU2) on this channel"); return
    audio = []; t_end = 0.0
    for k, c in enumerate(cl):
        algs = c["algs"]
        n80 = algs.get(0x80, 0); others = [n for al, n in algs.items() if al != 0x80]
        # single LDU2 readings can be wrong (no Reed-Solomon yet): clear when 0x80 is the majority, read at least twice, and no other algorithm repeats
        clear = n80 >= 2 and n80 > sum(others) and (not others or max(others) < 2)
        desc = ", ".join(f"{p25_voice.ALGO.get(al, hex(al))} x{n}" for al, n in algs.most_common(3)) or "no LDU2 seen"
        print(f"call {k + 1}: {c['start']:.2f} s, {len(c['ldus'])} LDUs ({len(c['ldus'])*0.18:.1f} s), encryption: {desc}" + ("" if clear else "  -> skipped"))
        if not clear:
            continue
        dec = imbe.Decoder(); pcm = []; f0 = []; vo = []
        for t, duid, frames in c["ldus"]:
            for bits in frames:
                pcm.append(dec.decode(bits))
                if dec.last and not dec.last["repeat"]:
                    f0.append(dec.last["w0"]*8000/(2*np.pi)); vo.append(dec.last["voiced"]/dec.last["L"])
        y = np.concatenate(pcm)
        print(f"   {len(pcm)} voice frames ({len(y)/8000:.1f} s): repeats {dec.repeats}, mutes {dec.mutes}; pitch median {np.median(f0):.0f} Hz (range {np.percentile(f0, 5):.0f}-{np.percentile(f0, 95):.0f}), "
              f"voiced share of harmonics {100*np.mean(vo):.0f}%, peak {np.abs(y).max():.0f}, rms {y.std():.0f}")
        gap = max(0, int((c["start"] - t_end)*8000)); audio.append(np.zeros(min(gap, 8000))); audio.append(y); t_end = c["start"] + len(y)/8000
    if a.wav and audio:
        y = np.concatenate(audio); pk = np.abs(y).max()
        y = y*24000/pk # level the whole file to a peak of 24000 (the vocoder's output is in 16-bit units: loud frames pass full scale, quiet calls are very low)
        print(f"scaled by {24000/pk:.2f} (peak was {pk:.0f})")
        y = y.astype(np.int16)
        with wave.open(a.wav, "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(8000); w.writeframes(y.tobytes())
        print(f"wrote {a.wav} ({len(y)/8000:.1f} s at 8 kHz)")


if __name__ == "__main__":
    main()
