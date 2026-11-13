"""P25 phase 2 voice to audio for calls that are not encrypted: one logical channel (slot) of a TDMA carrier -> half-rate vocoder (p25p2_voice.py) -> a WAV file

    python projects/p25_dmr/p25p2_audio.py data/survey/w852.7.cf64 --fs 4 --cf 852.7 --freq 853.5625 --wacn 0xbee00 --sysid 0x8d0 --nac 0x8d1 --wav out.wav

Per carrier: the slots are decoded (p25p2.py), each logical channel's calls are found from the MAC PTT / Group Voice Channel User messages and the ESS, and the voice frames of a call are decoded only when its algorithm
ID says unencrypted (0x80). Anything else is reported and skipped; no key is used or attempted. Output: 8 kS/s mono 16-bit WAV, written locally. The half-rate decoder has not been verified on speech (no clear
phase 2 call has been recorded yet). Meant to be upstreamed into RPPS.
"""
import argparse
import wave
from collections import Counter

import numpy as np

import p25p2
import p25p2_voice


def decode_file(path, fs, cf, freq, wacn, sysid, nac, seconds, ppm=15.0):
    x = np.fromfile(path, np.complex64, count=int(fs*seconds))
    z = p25p2.baseband48(x, fs, cf, freq*(1 + ppm*1e-6))
    off, mag = p25p2.carrier_offset(z)
    z = z*np.exp(-2j*np.pi*off*np.arange(len(z))/48000)
    dec = p25p2.Decoder(wacn, sysid, nac); last = -1
    for w0 in range(0, len(z) - 48000, 48000):
        seg = z[w0:w0 + 48400]
        A = [p25p2.symbols(seg, ph) for ph in range(8)]
        best = p25p2.find_slots(A)
        if best is None or best[0] < 0.8:
            continue
        sc, ph, r = best
        hits = [i for i in range(1, len(r) - 1) if r[i] > 0.8 and r[i] >= r[i - 1] and r[i] >= r[i + 1]]
        if hits:
            sl = dec.decode_slots(A[ph], hits[0], abs0=w0 + ph, after=last)
            if sl:
                last = sl[-1].t
    return dec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file"); ap.add_argument("--fs", type=float, required=True, help="[MS/s]"); ap.add_argument("--cf", type=float, required=True, help="[MHz]"); ap.add_argument("--freq", type=float, required=True)
    ap.add_argument("--wacn", type=lambda s: int(s, 0), required=True); ap.add_argument("--sysid", type=lambda s: int(s, 0), required=True); ap.add_argument("--nac", type=lambda s: int(s, 0), required=True)
    ap.add_argument("--seconds", type=float, default=30.0); ap.add_argument("--wav")
    a = ap.parse_args(argv)
    dec = decode_file(a.file, a.fs*1e6, a.cf*1e6, a.freq*1e6, a.wacn, a.sysid, a.nac, a.seconds)
    print("slots:", dict(dec.stats))
    audio = {0: [], 1: []}; calls = Counter(); state = {0: dict(alg=None, hv=None, tg=None), 1: dict(alg=None, hv=None, tg=None)}
    for s in dec.slots:
        if s.lch is None:
            continue
        st = state[s.lch]
        if s.pdu and s.pdu["ok"]:
            if s.pdu["name"] == "MAC_PTT":
                st.update(alg=s.pdu["algid"], tg=s.pdu["group"], hv=None); calls[(s.lch, s.pdu["group"], hex(s.pdu["algid"]))] += 1
            for m in s.pdu.get("msgs", []):
                if m["type"] == "group voice user":
                    st["tg"] = m["group"]
        if getattr(s, "ess", None):
            st["alg"] = s.ess["algid"]
        if s.voice and st["alg"] == 0x80:
            if st["hv"] is None:
                st["hv"] = p25p2_voice.HalfRate()
            audio[s.lch].append(np.concatenate([st["hv"].decode(c) for c in s.voice]))
    for k, v in sorted(calls.items()):
        print(f"slot {k[0]}: talkgroup {k[1]}, algorithm {k[2]}: {v} PTT message(s)" + ("" if k[2] == "0x80" else "  -> encrypted, not decoded"))
    for lch, parts in audio.items():
        if parts and a.wav:
            y = np.concatenate(parts); y = y*(24000/max(1.0, np.abs(y).max())); path = a.wav.replace(".wav", f"_slot{lch}.wav")
            with wave.open(path, "wb") as w:
                w.setnchannels(1); w.setsampwidth(2); w.setframerate(8000); w.writeframes(y.astype(np.int16).tobytes())
            print(f"wrote {path} ({len(y)/8000:.1f} s)")
    if not any(audio.values()):
        print("no unencrypted voice on this carrier")


if __name__ == "__main__":
    main()
