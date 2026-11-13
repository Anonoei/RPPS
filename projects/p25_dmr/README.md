# p25_dmr: P25 phase 1 / 2 and DMR decoding, IMBE voice

Difficulty: advanced. Needs a supported SDR, or a saved cf64 capture. Find the channels first with the tools in [../spectrum/](../spectrum/) (`vhf_scan.py`, `fsk_probe.py`). Listen-only: encrypted traffic is never followed, decoded or keyed.

## Decoders on a capture

    python projects/p25_dmr/p25_sync.py data/p25.cf64 --fs 4 --cf 407 --freq 407.6187 --seconds 20 --tsbk   # frame sync, BCH-decoded NAC / frame type, control-channel messages
    python projects/p25_dmr/p25_voice.py data/p25.cf64 --fs 4 --cf 407 --freq 406.8375                      # voice: encryption algorithm / key ID (LDU2), talkgroup (LDU1)
    python projects/p25_dmr/p25_audio.py data/p25.cf64 --fs 4 --cf 852.7 --freq 853.575 --seconds 90 --wav out.wav   # clear phase 1 calls -> WAV
    python projects/p25_dmr/dmr_sync.py data/dmr.cf64 --fs 8 --cf 462.5 --freq 462.0063 464.4187            # DMR bursts: sync kind, colour code, data type (raw slot type)

- `p25_sync.py --tsbk` also reports TDMA channel tables (IDEN_UP_TDMA) and the slot of each voice grant (a control channel that announces them means phase 2), and each talkgroup's service options (bit 0x40 = encrypted).
- `imbe.py` is a full-rate IMBE decoder written from TIA-102.BABA (tables in `imbe_tables.py`, parsed from that document); `p25_audio.py` feeds it the voice frames of the LDUs. Calls whose LDU2s say anything but algorithm 0x80 (unencrypted) are skipped. Output stays on disk.
- `p25_voice.py` applies the LDU Reed-Solomon codes (`rs/p25_24_12`, `rs/p25_24_16`); the codes themselves are RPPS definitions.

## p25_monitor.py: live P25 phase 1 monitor in the browser

    python projects/p25_dmr/p25_monitor.py --cf 409.0 --fs 4 --control 407.6125 --gain 1 32 16     # open http://127.0.0.1:8025/
    python projects/p25_dmr/p25_monitor.py --file data/w409.0.cf64 --cf 409.0 --fs 4 --control 407.6125   # replay a recording in real time

One radio covers the control channel and the voice channels of the system (they must sit inside `--fs`, about 80% of it usable). The control channel's grants list the talkgroups and their voice frequencies; a decoder follows every granted frequency, so a call is caught from its first frame. The page lists talkgroups as clear, encrypted (grant flag or algorithm ID) or phase 2, and plays a chosen talkgroup, or scans all clear ones. Encrypted talkgroups are never followed or decoded. `--ppm` is the radio's clock error (default 15). Binds 127.0.0.1 only; audio is kept in memory, not saved.
The page also shows the radio's tuning (live spectrum with the decoded channels, centre, decodable span, gain), a log of grants, calls and encryption decisions, the control channel's message counts, and the last clear calls for playback. "Test sound" checks the browser's audio path without the radio.
With the radio on another machine: `--remote tcp://host:5555` (or `PYSDRLIB_REMOTE`, see [../README.md](../README.md)).

## Phase 2 (TDMA): p25p2.py, p25p2_voice.py, p25p2_audio.py

    python projects/p25_dmr/p25p2.py data/p25p2.cf64 --fs 4 --cf 852.7 --freq 853.5625 --wacn 0xbee00 --sysid 0x8d0 --nac 0x8d1
    python projects/p25_dmr/p25p2_audio.py data/p25p2.cf64 --fs 4 --cf 852.7 --freq 853.5625 --wacn 0xbee00 --sysid 0x8d0 --nac 0x8d1 --wav out.wav

Written from TIA-102.BBAB / BBAC (burst layout in Annex E, scrambling, ISCH and DUID codes, the Reed-Solomon (63,35,29) codes over GF(64), the MAC PDUs) and BABA-1 (half-rate vocoder), not from another decoder.
The H-DQPSK carrier is differentially demodulated, slots are found from the inter-slot sync, descrambled with the system's WACN / system ID / colour code, and the MAC PDUs, ESS and voice frames decoded.
Signalling slots are checked by Reed-Solomon and CRC-12. The half-rate vocoder's output is **unverified**: two readings of the draft addendum are noted in `p25p2_voice.py` (the pitch table lists w0 / 2 pi; Eq. 37 is read as C(i,k) = H(i,k-2) up to k = 6). Tone frames are not implemented.
`p25_monitor.py` follows phase 2 carriers from the control channel grants and shows both slots of each (talkgroup, source unit, algorithm).
