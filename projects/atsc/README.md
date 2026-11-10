# atsc: ATSC (8VSB) digital TV

Run from the repo root. [ATSC.md](ATSC.md) has the full story: the method, the parameters, the traps, the planned scan/probe/play feature and what should become part of RPPS.

## atsc_monitor.py: the live monitor in the browser (real time)

    python projects/atsc/atsc_monitor.py                                   # then open http://127.0.0.1:8035/
    python projects/atsc/atsc_monitor.py --remote tcp://HOST:5555 --channels 19 20 35
    python projects/atsc/atsc_monitor.py --replay data/atsc/atsc35_6s.cf64 --replay-channel 35 --replay-cf 604 --replay-speed 0.5   # a capture as the air (it loops: the wrap loses lock once per loop)

Like `p25_dmr/p25_monitor.py`: a small HTTP server (127.0.0.1) and one page. The sweep finds the stations (pilot level) and probes each (SNR on the segment syncs, Reed-Solomon yield, PSIP names and streams); pick a program and
press Watch: the radio goes to that channel, the streaming receiver decodes it in real time, ffmpeg re-encodes the program (deinterlace, H.264 / AAC, fragmented MP4) and the page plays it through Media Source Extensions
(`/video`, ~3-5 s behind). While watching it shows lock state, SNR, Reed-Solomon yield, symbol-clock error, seconds behind real time, per-thread load, bit rate, the channel's spectrum and an SNR / yield history.
The receiver is `atsc_rx.py` (`Receiver.feed()` / `.start()` + `.push()`, `decode_array()` for a finished capture) on `atsc_front.py` (the whole front end in one C pass: mix, filter, resample to one complex sample per symbol).
The page plays `/video`; `ffmpeg -i http://127.0.0.1:8035/video out.mp4` while watching shows whether the server side is fine.

## atsc_live.py: chunked tools: list, watch, curses monitor (older, not real time)


    python projects/atsc/atsc_live.py                          # curses monitor: up/down, Enter = programs, w = watch, x = stop, r = rescan, q = quit
    python projects/atsc/atsc_live.py scan                     # pilot level of every channel, once
    python projects/atsc/atsc_live.py list 19                  # decode a channel, print its virtual channels (PSIP names) and streams
    python projects/atsc/atsc_live.py watch 19 20.1            # play virtual channel 20.1 in ffplay (--player vlc | stdout | file:out.ts)
    python projects/atsc/atsc_live.py --remote tcp://host:5555 ...   # a pysdrlib server's radio (or $PYSDRLIB_REMOTE)
    python projects/atsc/atsc_live.py --replay data/atsc/atsc35_6s.cf64 --replay-channel 35 --replay-cf 604 list 35   # no radio: a capture as the air

The monitor sweeps the channels (pilot over the data around it: an FFT), then probes each station in worker processes (2 s capture, lock, SNR from the segment syncs, Reed-Solomon yield, PAT / PMT / PSIP) and keeps
refreshing. Status: OK (>= 90% of codewords corrected), marginal, weak, no lock. `watch` captures continuously, decodes overlapping ~2 s chunks in a process pool, numbers every field from its time so the overlaps are
dropped and the chunks join, keeps the packets of the chosen program and paces them to the player at the stream's rate.
Speed: the trellis and Reed-Solomon stages are `rp.coding` (`trellis/atsc_8vsb`, `RS(207, 187)`, run by the `rpps._native` C++ kernels when built, `RPPS_NO_NATIVE=1` for the Python loops; `atsc_decode.rs_decode` batches the codewords), `atsc_probe` trains each equalizer on at most 120 000 symbols and works in blocks.
**It is not real time on a laptop yet**: a 2 s chunk costs ~12 CPU-seconds and, being memory-bound, 8 workers decode about 0.4 s of air per second (measured on an 8-core Ryzen). `watch` therefore skips chunks (a gap)
when the pool is behind, and the stream is only continuous when the air is slower than that (`--replay-speed 0.35` is: the output decodes to 1080p in ffmpeg). What is left is in ATSC.md ("Getting to real time").
Remote radio notes: after a retune the pipeline holds ~0.25 s of the old frequency (`Radio.tune` flushes 0.45 s), and a few blocks per second are dropped on the link (`dropped blocks` in the status): a chunk with a gap does not lock.

## atsc_probe.py: look at an ATSC (8VSB) TV channel

    python projects/atsc/atsc_probe.py data/atsc19.cf64 --fs 20 --cf 507 --lower 500 --plot atsc.png

Finds the pilot (and so the carrier and clock offset), locks to it, recovers the symbol clock from the segment syncs,
slices the eight levels, finds the field syncs, and trains an equalizer on them. Reports MER before and after the equalizer
and the slicer error rate. Tune so the radio's DC spike is outside the 6 MHz channel (it is at the capture's centre).

## atsc_decode.py / atsc_diag.py: ATSC to a transport stream, and its diagnostics

    python projects/atsc/atsc_decode.py data/atsc19.cf64 --fs 20 --cf 507 --lower 500 --out ch19.ts --save-soft data/atsc19_soft.npy
    python projects/atsc/atsc_diag.py layout --soft data/atsc19_soft.npy      # one check per assumption: relation, levels, edges, rotation, ...

The decoder runs the trellis (Viterbi), byte layout, de-interleaver, Reed-Solomon (rp.coding.RS(207, 187)), derandomizer and
writes the MPEG transport stream. The diagnostics are what found each problem on the way (every one prints a number that is ~50%
when its assumption is wrong). **projects/atsc/ATSC.md** has the story, the parameters, the traps, and what should become part of RPPS.

The equalizer is trained per block of `atsc_probe.BLOCK_FIELDS` (10) fields, not once per capture: the channel changes during a capture, and the taps trained on 6 s of channel 35 differed 28% from those of its first second.
With one equalizer the Reed-Solomon failures piled up (76% of blocks correctable over 6 s, with the start of the capture at ~100% bad); with 10-field blocks it is 97.6% (40-field blocks: 89%). Sample-clock drops are not the cause: the segment-sync
position track has no jumps. `recover(..., block=0)` gives the old single equalizer; `atsc_decode.py --chunk` (decode in pieces) is only for bounding memory now.

## atsc_scan.py: find the stations in your area

    python projects/atsc/atsc_scan.py                       # UHF 14-36 and VHF 7-13, writes data/atsc/ch<N>.ts
    python projects/atsc/atsc_scan.py --channels 15 19 35 --seconds 4 --out-dir data/atsc

Per channel it tries a few gains with a short capture, keeps the one with the best equalized MER, and decodes the channels that lock (`atsc_probe.recover`, `atsc_decode`) into a transport stream, then lists the programs with
`ts_info.py`. A channel where `recover` raises is reported on stderr with the gain and the error, then skipped.

## ts_info.py: what is in the transport stream

    python projects/atsc/ts_info.py ch35.ts      # programs (PAT / PMT), their streams, and the PSIP channel names (virtual channel table)

Capture note: put the tune centre outside the 6 MHz channel (the radio's DC spike), and check the channel's lower edge with the sweep (channel n spans 470 + 6 x (n - 14) MHz, so 596-602 MHz is channel 35).
A few seconds are needed for a video frame (`ffmpeg -i ch35.ts -map 0:v:0 -frames:v 1 frame.png`): MPEG-2 sequence headers come once per GOP.
