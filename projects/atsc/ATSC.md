# ATSC 8VSB receive chain: notes, diagnostics, and what belongs in RPPS

Working notes from bringing up an ATSC decode from a radio capture (UHF ch 19, 500-506 MHz). The code is in `projects/atsc/`:

| file | what |
|---|---|
| `gain_search.py` | pick the gain by SNR, penalising clipping and gain-made spurs |
| `record.py` | record a capture to complex64 |
| `atsc_probe.py` | pilot, carrier and clock offset, symbol clock from segment syncs, equalizer, field syncs. `recover()` returns soft symbols |
| `atsc_decode.py` | trellis (Viterbi), byte layout, de-interleave, Reed-Solomon, derandomize, transport stream |
| `atsc_diag.py` | **the diagnostics**: one check per assumption, each prints a number that is ~50% when the assumption is wrong |
| `rds_link.py`, `fm_rx.py` | the FM receiver (a separate story) |

## The method that worked

A decoder with many unverified assumptions fails silently: "0 good codewords". Don't change several things and re-run.
For each assumption find a statistic that is 50% (chance) if it is wrong and ~90%+ if it is right, and test it alone.
Reed-Solomon is the final judge: a valid codeword happens by chance with probability ~3e-7, so *any* correctable
codeword proves everything before it is right, and the *fraction* correctable measures how right.

Use `atsc_diag.py <check> --soft <cache>` (write the cache with `atsc_decode.py --save-soft`).

## What each check caught (in the order it was needed)

1. **`relation`: is the trellis structure right?** The 4-state code satisfies `Z0[n] = Z0[n-2] xor Z1[n-1]`. Holds ~91% on hard
   decisions; every alternative is exactly 50%. This validated the code and the level -> bit mapping (k = 4*Z2 + 2*Z1 + Z0,
   level = 2k - 7) from the data, instead of from memory of the standard.
2. **`levels`: gain and offset.** The soft values were compressed 7% (centres at 0.928 x level) because gain had been calibrated
   from the sync symbols, which an equalizer leaves biased by ISI. Symptom: outer levels hold 8% / 11% instead of 12.5%, and
   the slicer mislabels them. Fix: fit gain/offset to the data's own eight clusters (`atsc_probe.fit_levels`).
3. **Clock drift.** Stopping the clock-recovery loop at "< 0.5 ppm" left a 0.05 ppm error, which over a 0.6 s capture walks the
   symbols by 0.3 of a symbol. Raw MER went 11.3 -> 16.7 dB when it was iterated to < 0.003 ppm. Use a regression over all
   the sync positions, not their endpoints.
4. **`edges`: errors only at segment edges.** Viterbi disagreed with the slicer on 24 / 14 / 8 / 4 / 3 / 2 % over the first
   steps of every segment, mirrored at the end (a decaying edge effect = the decoder believes each segment is a separate path),
   while the noise power was flat. So the symbols were fine and the *pairing* of encoder streams across segments was wrong.
5. **`rotation`: how the 12 encoders advance per segment.** Relation across a segment boundary for each rotation d: 91.7% at
   d = 4, ~50% for the other eleven. The 12 encoders advance through all **832** symbol slots of a segment, the 4 uncoded
   sync symbols included, and 832 = 4 (mod 12). So data symbol t of data segment s belongs to encoder (4s + 8 + t) % 12.
   This is the trellis PATH pairing only: the standard says the same (5.2.4: the input demultiplexer and output multiplexer
   "shall advance by four positions on each segment boundary"). It is NOT how bytes are laid out (next item).
6. **`rs-errors`**: maps corrected bytes back to where they sat in the received stream: by position in the segment (edge
   problem), by deinterleaver branch n % 52 (interleaver delay problem), by encoder (one encoder wrongly paired).
7. **`layout`**: byte-assembly hypotheses, each judged by Reed-Solomon on the same Viterbi output. Three of the four were wrong
   (0 of 1508); **C: in blocks of 12 segments, four consecutive steps at one position t % 12 are byte 12*row + position** gave
   1507 of 1508. Note it differs from the path pairing above: paths follow the slot rotation, bytes follow position. (An earlier
   384-way search over bit order / precoder / byte phase / offset found nothing, because every variant sat on top of the wrong
   per-encoder layout. A search over secondary choices is worthless if the primary structure is wrong: change the structure first.)
8. **`randomizer`**: keystream xor observed byte must be one constant at every packet start. The standard's randomizer passes
   (100%); the same register shifted 8 times per byte gives 1.6% (chance).
9. **`viterbi-selftest`**: encodes random data with the assumed trellis, adds noise, decodes. Validates the Viterbi code on its own
   (0 errors at sigma 0.47; 0.17% dibit errors at sigma 0.8, i.e. 15 dB). If this passes and real data fails, the assumed code
   differs from the real one: look at `relation` / `rotation`, not at the decoder.

## When stuck: read the standard (it was the answer)

Hours went into guessing the randomizer from memory (192 LFSR variants x strides x offsets, all at chance). Reading A/53 part 2
figure 5.3 solved it in minutes: a **Galois** register (feedback added after stages 1, 3, 6, 7, 11, 12, 13), preload 0xF180,
and the randomizing byte read **in parallel from stages 1, 3, 4, 7, 11, 12, 13, 14 (D0..D7)**, the register shifted **once per
byte**, over the 187 data bytes only. A serial-output LFSR family could never have matched. The text of 5.2.4 also states the
rotation of 4 that `rotation` had measured.
How to read a standard that is an encrypted PDF: the web fetch tool can't parse it, but `pdftotext -layout file.pdf out.txt`
(poppler) can, and `pdftoppm -f N -l N -r 300 -png file.pdf out` renders a page so a figure can be read. Source:
https://www.atsc.org/wp-content/uploads/2015/03/a_53-Part-2-2011.pdf (section 5.2.1 and figure 5.3; 5.2.4 and 5.2.5 for the
trellis encoders and the intra-segment interleaver). Next time: read the relevant section BEFORE writing the stage.

## Traps (things that looked like results and weren't)

- **"Slicer error rate"** computed as `|soft - nearest level| > 1` is NOT a symbol error rate. Any value is within one level of
  *some* level, so it only counts values beyond the outermost levels. An estimate from MER is the honest one: at 19.6 dB MER
  (sigma ~0.48) about 3% of hard decisions are wrong. I reported 0.98% and "decodes below ~4%": both wrong.
- **Averaging power across a band** is dominated by the radio's DC spike (and spurs at multiples of 20 MHz). Mask +-150 kHz
  around the tuned frequency and use the median.
- **Peak-above-floor of a narrow tone** needs a floor measured in the same bin width; averaging a band dilutes it.
- **The wrong test passes in the wrong place**: the relation held 91% *inside* segments while failing across them, because the
  first version of the check never crossed a boundary. Test across boundaries.
- A **good-codeword count on a small subset** can read zero for the right answer when the right answer is only ~3% good. Size
  the test to the expected rate.
- **A search over the wrong family**: serial-LFSR variants scored chance on every test, which says "none of these", not "the
  test is broken". When a whole family scores chance, question the family.
- Python `cd dir && python - <<EOF` silently skips the Python when the `cd` fails (the edit never applied, and the old code
  ran). Use absolute paths.

## Parameters worth remembering

ATSC: symbol rate 4.5e6 * 684/286 = 10.7622 MBd; pilot 309.44 kHz above the channel's lower edge; roll-off 0.1152; segment 832
symbols (4 sync + 828 data = 207 bytes); field 313 segments (1 field sync + 312 data); RS(207,187) t = 10, GF(256) 0x11D, roots
alpha^0..alpha^19; convolutional byte interleaver B = 52, M = 4 (delay 10608 bytes), restarted each field (312 * 207 = 0 mod 52).

## What should become part of RPPS

Not done; for later. In rough order of reuse:

1. **`rp.mod.VSB` / `rp.sync.ToneLock`**: an 8VSB demodulator stage (pilot lock, matched filter, symbol clock from segment sync) and
   a generic pilot-lock stage (`ToneLock` in `atsc_probe.py`, and `PilotPLL` in `fm_rx.py`, are the same idea).
2. **`rp.filters.equalizer`**: least-squares / decision-directed equalizer (`train_equalizer`, `apply_equalizer`), which is general.
3. **`rp.coding` trellis code**: the ATSC 4-state code with an uncoded precoded bit and its Viterbi decoder (`viterbi`), batched over
   interleaved encoders. The decoder is vectorised across encoders and fields (~2 s per field batch).
4. **`rp.coding` convolutional byte interleaver** (`deinterleave`) and an `rs/207_187` config (the existing RS handles it: it needs
   only `config/coding/block/rs/207_187.json` = n 207, k 187, poly 285, fcr 0, prim 1).
5. **`rp.analysis` diagnostics**: the `atsc_diag.py` pattern, "one statistic per assumption", in a form that works for any
   framed/coded signal (relation-holds-% for a code, boundary-pairing scan, layout-variant search against a checksum).
6. **Signal-quality helpers** (`fit_levels`, `mer_of`, an honest raw-symbol-error estimate from MER).
7. **Radio control** (`gain_search.py`) belongs in pysdrlib, not RPPS; `rp.serial.Device` already needed a fix (busy-polling
   `get_samples`) that is in `src/rpps/serial/source.py`.

## Status

**ATSC ch 19 decodes to a transport stream** (a 0.6 s capture, 22 whole fields):

| stage | result |
|---|---|
| Reed-Solomon | 7123 of 7124 segments correctable; mean 8.1 bytes corrected each (limit 10; 159 at the limit: thin margin) |
| transport stream | 7124 packets; continuity counters in order for 6891 of 6891 (100%) |
| multiplex | 5 programs: ch 3 HD (MPEG-2 1280x720, 59.94 fps, AC-3 stereo), programs 4-7 SD (720x480) with AC-3 |
| picture | ffmpeg decodes frames from the HD and from two SD streams; HD frame: a college football broadcast |

Not done: audio playback was not tried; the capture is short, so most streams have no sequence header yet (use a few seconds);
the signal is marginal (MER 20 dB, RS near its limit), so a longer capture or a better antenna/gain would help; ch 27 (14.3 dB
MER) and ch 22 were not decoded; PSIP (station names, program guide) was not parsed. Only one capture has been decoded, so
the parameters above are verified on one station.

Reproduce:

    python projects/common/record.py 507 20 3 data/atsc19.cf64 --gain 0.8
    python projects/atsc/atsc_decode.py data/atsc19.cf64 --fs 20 --cf 507 --lower 500 --out ch19.ts --save-soft data/atsc19_soft.npy
    ffplay ch19.ts            # or: ffmpeg -i ch19.ts -map 0:3 -frames:v 1 frame.png


## Planned, not built: scan channels, probe each, play one live

The request: scan for ATSC signals, scan each channel for its programs, then pick a signal and a program and play it live.
Nothing below exists yet; it is the design and the measurements it rests on.

### The three commands

1. **`scan`**: step across the TV bands (US: VHF-low 2-6 = 54-88 MHz, VHF-high 7-13 = 174-216, UHF 14-36 = 470-608, up to 51 = 698;
   the radio covers all of them) with 20 MS/s captures, mask the DC spike (+-150 kHz at each tile centre) and the 20 MHz-multiple
   spurs, and call a 6 MHz slot ATSC when it has a flat plateau and the pilot 0.309 MHz above its lower edge (pilot 15-35 dB over the
   data was typical here). Report channel number, frequency, pilot level, plateau SNR. Cheap: FFTs only (the first UHF sweep took
   about 40 s for 476-700 MHz). Gain per band as the standing preference says: `gain_search.py` per channel is ~4 min at the full
   grid, so use a reduced grid (amp x LNA 16/24/32/40 x VGA 16/24/32, ~24 points) once a plateau is seen.
2. **`probe CH`**: record ~2-3 s (PSIP's VCT repeats every ~400 ms, PMTs every ~100 ms), decode to a transport stream, and parse
   PAT / PMT / PSIP (PID 0x1FFB; VCT tables 0xC8 / 0xC9 give the major.minor numbers and short names) into a program list: number,
   name, video format, audio. The decode is the same code as `atsc_decode.py`. ffprobe already gave programs, codecs and sizes
   from our capture, so a first version can shell out to it; PSIP names need our own parser (ffprobe did not show them).
3. **`play CH [PROGRAM]`**: tune, decode continuously, demux the chosen program and pipe the TS to ffplay/VLC (both installed here).
   Needs real time, which the current code is not (below).

### What the code costs today (measured, 0.6 s of air, 20 MS/s, one core)

| stage | time | per second of air |
|---|---|---|
| `atsc_probe.recover()` (all of demodulation, equalizer training and two decision-directed passes) | 11.0 s | 18 s |
| of which `train_equalizer` (3 calls) | 4.3 s | 7 s |
| of which `np.convolve` (21 calls: matched filter and equalizer, direct convolution) | 2.3 s | 4 s |
| of which `fit_levels` / `polyfit` | 2.0 s | 3 s |
| rp pipeline (ShiftFreq, 201-tap low-pass; stateful, block-wise) | 1.8 s | 3 s |
| Viterbi (`viterbi`, 24 fields batched: 288 streams) | 2.0 s | 3.4 s |
| Reed-Solomon (`rp.coding.RS(207,187)`, per-codeword Python correction; 99.99% of codewords need it) | 24 s | 40 s |

Machine: 16 cores, 46 GB, scipy 1.16 installed but **numba is not** (and scipy is not an RPPS dependency). So today the whole
chain is ~60x slower than real time, almost all of it Reed-Solomon, then demodulation, then Viterbi.

### Why live is feasible anyway (what to build)

- **Calibrate once, then stream.** Everything expensive in `recover()` is estimation that does not change within a session: pilot
  frequency, clock ratio, segment alignment, equalizer taps (the channel is static). Estimate them from a ~1 s capture, then run a
  fixed pipeline with only tracking loops: `ToneLock` for the pilot, the sync-position drift for the clock (a slow correction of
  the resampler step), periodic equalizer refresh from the field syncs.
- **Fold the filters.** Matched filter (33 taps) and equalizer (96 taps) are linear and adjacent: one ~128-tap filter. Use FFT
  overlap-save instead of direct convolution (numpy `rfft`; no scipy needed): direct convolution of 21.5 MS/s x 96 taps is ~2 GMAC
  per second of air. The 201-tap complex low-pass at 20 MS/s the same way.
- **Viterbi: batch big.** The loop is over time steps (21 528 per field) with a vector over streams; per-step cost is mostly numpy
  call overhead, so more streams per call is nearly free. 24 fields (0.58 s) took 2.0 s; 200 fields (4.8 s of air) should be under
  real time. The price is latency of a few seconds.
- **Reed-Solomon: write a batched decoder.** Syndromes are already vectorised; the Python cost is Berlekamp-Massey, Chien search
  and Forney per codeword. All three vectorise across codewords (BM is 20 fixed iterations; Chien is a (207 x 10) evaluation per
  word; Forney is the same). Expect 100-1000x. This also belongs in `rp.coding.RS` (it benefits DVB-S too).
- **Continuity across chunks.** The byte deinterleaver has 10 608 bytes of state, restarted per field; chunk at field boundaries
  and carry the last 51 segments' bytes forward. The trellis restarts per field so fields are independent (that is why batching them
  works).
- **Process layout.** One thread captures (the radio must be drained; see `fm_rx.py`'s reader thread and queue), N worker
  processes demodulate/decode field batches, one assembler orders them and feeds the player. Expected latency 5-15 s; that is
  "live" for a TV channel, not for interactive use.
- **Sample-clock drift to the player.** Same problem as `fm_rx.py` (radio clock vs the player's): the TS carries its own PCR, so
  let ffplay/VLC handle it, and keep the pipe fed above real time.

### Open questions to settle first

1. Is the signal good enough to play continuously? On ch 19 Reed-Solomon averaged 8.1 corrections per codeword with 159 of 7124 at the
   10-byte limit, so a long capture will have uncorrectable segments (glitches). More gain/antenna work, or a better equalizer
   (the one used here is trained only on field syncs + decision-directed on one 0.6 s capture), is needed before judging.
2. Only one station has been decoded. Parameters in this file are verified on ch 19 only; the scan needs to be run on every
   channel the antenna can see, and ch 22 / ch 27 (weak) are the stress tests.
3. Audio: AC-3 stereo; ffmpeg/VLC decode it. Not yet tried.
4. PSIP parsing is new code (MGT 0xC7, VCT 0xC8/0xC9, EIT 0xCB/0xCC, ETT 0xCC/0xCD).

### Suggested order

(a) batched Reed-Solomon, (b) `scan` (FFT only), (c) `probe` using the present offline decode plus a PSIP parser, (d) calibrate-once
streaming demodulator with folded FFT filters, (e) big-batch Viterbi and the chunked assembler, (f) `play`.


## Built: live monitor, list, watch (`atsc_live.py`) and what it measured

`atsc_live.py` now does the scan / list / play of the design above (see README.md). Measured on the way:

- **Throughput.** Reed-Solomon was the cost (5.6 s per second of air); `rp.coding.RS` through `rpps._native` (`atsc_decode.rs_decode`; it replaced the scripts' own `rs_batch.py`, same output) makes it 0.2 s. The generic numpy Viterbi
  held a (T, W, S) int64 table (1.4 GB for 83 fields); `rp.coding` fsm now runs the same decoder in C++ (`rpps._native.viterbi_batch`; it replaced `fastvit.py`). `recover` trains on a subsample (same MER and Reed-Solomon yield) and its BLAS threads must be pinned
  (`OMP_NUM_THREADS=1`: twelve workers with threads cost 63 CPU-s per chunk instead of 18). After that one 2 s chunk is ~12 CPU-s, a flat profile (resample 1.2, convolutions 1.4 + 1.5, training 0.9, fit_levels 1.3,
  pilot lock 1.4, Viterbi 0.8, RS 1.4); 8 workers give 0.4x real time because the chunk is memory-bound (2.2 GB peak). Real time needs the "calibrate once, then stream" demodulator above (one resample pass, fixed equalizer
  taps refreshed per chunk, float32 throughout): roughly 5 CPU-s per chunk.
- **Stitching.** Chunks overlap by 3 fields; the Reed-Solomon codewords of a chunk are numbered `field * 312 + j` from the time of its first field sync (radio clock, rounded against the previous chunk, so a 15 ppm clock error never
  accumulates). Stream from a 6 s capture replayed at 0.35x: 110 187 packets, 98% continuity counters in order, the breaks being isolated uncorrectable packets (the capture's 2% Reed-Solomon failures), not chunk joins; ffmpeg decodes 1080p.
- **`mer_eq` flatters.** A decision-directed equalizer reports 17 dB on a remote capture that decodes 0% of codewords; the segment-sync noise (SNR column) said 10-12 dB. Judge by SNR and the Reed-Solomon yield.
- **Remote radio.** After `set_rx_freq(..., flush=True)` the first ~0.25 s are the old frequency; a retune followed by a short capture gave "no field sync" until the flush was added.


## Built: the streaming receiver (real time) and the browser monitor

The chunked `atsc_live.py` above runs at ~0.4x real time because `recover()` re-estimates everything from every capture. `atsc_rx.py` estimates once and tracks (`atsc_monitor.py` is the front end, README.md):

- **One complex sample per symbol.** The channel centred on 0 is a complex signal 6 MHz wide, so 10.76 MS/s complex holds all of it; the real T/2 signal at 21.5 MS/s of `atsc_probe` is not needed. `atsc_front.py` does mix + low-pass +
  fractional resample in a single C pass (polyphase windowed sinc, 512 phases, 0.15 CPU-s per second of air; the numpy chain was ~3). A complex 64-tap symbol-spaced equalizer has the same two real degrees of freedom per symbol as
  the 96-tap real T/2 one; its output is `Re(phasor_n * sum g_m z[n+m-24]) + bias`, the phasor making the pilot real (pilot phase per 8192 symbols, applied per 1024).
- **Acquire once, track.** Pilot (FFT) -> clock (sync drift, as `recover`, but re-acquisition reuses the last clock: 2 passes) -> segment / field sync -> equalizer from the known symbols (field-sync PN511, every segment sync), then 2 decision-directed
  passes. Tracking per 4 fields (~97 ms): carrier phase, equalize, re-solve the equalizer on this region's decisions (normal equations, forgetting 0.2: the channel moves, so memory hurts: 0.98 gave 81% vs 96%) and equalize again, trellis,
  de-interleave (history carried), Reed-Solomon, de-randomize. The segment-sync position of the *unequalized* signal drives the symbol clock (gain 0.25); an equalizer cannot do it (it absorbs fractional delay).
- **Things that broke it, in order.** (1) `p` was a float32 scalar, so `step *= 1 + ...` made the clock step float32 (0.06 ppm = half a symbol per second of drift): lock lasted 1 s. (2) The front-end thread ran seconds ahead of the region thread, so the
  clock loop's corrections arrived late and it oscillated (bound the queue between them: 4 items). (3) A dropped region breaks the de-interleaver history, so the decode queue is unbounded. (4) `discardcorrupt` in ffmpeg threw away every frame of a
  stream with a few damaged macroblocks. (5) The input queue was sized in blocks assuming 1M-sample blocks; a remote radio sent 131k-sample blocks, so it held 0.7 s instead of 6 and every hiccup became a gap (and each gap a re-acquisition, 3 s of
  compute, more backlog: a cascade). Probes of other channels still running when a watch started took two cores.
- **Cost per second of air.** C Reed-Solomon (`rs_batch.py`, 3.6 us a codeword; syndromes as 20 independent table chains per byte) 0.05, C Viterbi (`fastvit.py`, 8 streams per SIMD group) 0.35, regions 0.6 (the normal equations are a BLAS matmul:
  set OMP_NUM_THREADS ~3), front end 0.15. Three threads overlap them because every heavy call releases the GIL.
