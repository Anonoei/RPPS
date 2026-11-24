# Changelog

RPPS is 0.x: a minor version may rename or remove things, and old names are not kept as aliases (except the stable API
in the Building on RPPS guide, which warns for one release). Every release lists what breaks under **Breaking**, with
what to use instead. Work in progress (not yet here) is in [WORKING.md](WORKING.md).

## Unreleased

### Breaking
- **`rp.analysis.catalogue.SYSTEMS` is gone: the systems are `config/system` definitions**, one file each
  (`extends`, `abstract`, tags and user config layers work as for any definition); `catalogue.systems()` loads them.
  `match` now reports systems the features can't tell apart as `tied` (ambiguous) instead of ordering them by name,
  and gives no positive score on weak evidence (a frequency alone, a modulation alone): two agreeing features, or one
  strong one (baud, chip_rate, scs, code_length, profile, subcarrier, or an unmodulated carrier).
- **`{"signal": "stage"}` is `{"signal": "class"}`**, and every category takes a `class` step: your own operator by
  module path (`"class"`, `"args"`), a Stage for signal, sync and wave (`"tx_class"` for the transmit side), a Pipe for
  symbol, code and link (decoding on receive, encoding on transmit). A class whose CATEGORY is another category's is
  refused.
- **BLE, IEEE 802.11b and 802.15.4 payload bytes are in the standards' bit order.** Those standards send each octet
  least significant bit first; `ble/adv` and `ieee80211b/long` sent (and read) their payloads most significant bit first,
  so a real packet's payload came out bit-reversed per byte (its CRC still passed: it is over the bits as sent), and
  `ieee802154/2450` sent each octet's high nibble first. Bytes in and out are unchanged in meaning: what is on air now
  matches the standards. A link header's new `"payload_order": "lsb"` does it for `ble_adv` and `ieee80211b_plcp`; the
  `interleave/block/lsb_first` definition (each octet's bits reversed) is the last step of `ieee802154/2450`, whose
  receive output is now data bits (Rep.DATA), not channel bits.
- **`fec/block/trellis/p25_1_2` is one codeword per 96 bits** (its new `length`): encode and decode take whole
  codewords, each started and flushed on its own (96 bits to 196), so it works as a profile's block step. A whole
  stream decoded as one trellis sequence is `decode_batch` on a row.

### Added
- **Burst time plans** (`rpps.plan`, a new `plan` resource): TDMA / MF-TDMA plans in DVB-RCS2's model (superframe
  sequences, frames, a BTU grid of carriers and timeslots, burst types, assignments), with frequency hopping (`hop`: an
  expression of the superframe and frame, or an algorithm: `ble_csa1`) and trains (`repeat`, `every`). From a definition,
  from a forward link (`rp.plan.builder("dvb-rcs2")`: SCT, FCT2, BCT, TBTP2 and the NCR clock, read by the new
  `profile/dvbrcs2/forward`; `rp.plan.rcs2.tables` writes them), or learned from detected bursts
  (`rp.analysis.spectrum.burst.learn_plan`, saved as a definition).
- **Bursts on many carriers** (`rp.sync.Bursts`; `{"sync": "burst", "carriers": [...] | "plan": ...}`): each carrier
  channelized (to `sps` samples a symbol), its neighbours filtered out, cut at absolute positions; with a plan each burst
  is placed (superframe, frame, slot, tx type, assignment), the plan's origin acquired from the bursts and tracked from
  their unique words, `gate: "plan"` cutting slots energy cannot see. A burst's fields drive the steps after it and ride
  on its messages (`fields["burst"]`). `rp.sync.PlanTx` places bursts by a plan; a profile's transmitter does it from
  each burst's fields.
- **GSE** (`{"link": "gse"}`, `rp.link.gse.pack`): PDUs in DVB-S2 data fields, fragmented and reassembled (CRC-32),
  each PDU a frame of its own for the steps after it.
- `profile/demo/mftdma` (three carriers, two rates, three burst types, LDPC units) and `examples/mftdma.py`: a network
  whose return link is decoded with the plan its forward link gives. Definitions: `header/link/rcs2_table`,
  `sequence/word/demo_uw1..3`, `table/demo/mftdma_types`, `plan/demo/mftdma`; tag `dvb-rcs2`.
- `FrameSync(count=)`: frames to find in a stream (a burst's one); `{"sync": "timing", "method": "oerder_meyr",
  "clock": false}`: the phase only (a burst a few windows long); the code marker's `sequence` is a controlled parameter.
- **Codes bit by bit:** any block code takes `append` (check bits after the codeword, each the XOR of the codeword bits
  listed), `extend` (an overall parity bit, now for any block code), `layout` (where each codeword bit is sent, or
  `{"message": [...], "parity": "rest"}`) and `xor` (a pattern on the sent word), decoded in reverse, soft or hard
  (`rpps.code.types.layout`). A `linear` code takes its parity bits as equations (`"parity": [[0, 1, 3], ...]`), and the
  new `expr` type each codeword bit as an expression of the message bits: loaded as a linear code in its own layout when
  it is affine, decoded from its codebook otherwise. `fsm` trellis codes take a codeword `length`. New:
  `bch/p25_nid` (BCH (63,16) and its DUID bit, checked against TIA-102.BAAA-A's generator matrix), which
  `projects/p25_dmr/p25_sync.py` now loads.
- **Frame sync by named words, among symbols** (`{"sync": "frame"}`, `rp.sync.FrameSync`): alternatives (`sequences`:
  {name: word}; `zc`: a set of Zadoff-Chu roots and shifts), the one found published as `word` with its fields; a word
  mid-burst (`before`, `length`, controlled); real levels kept real, a polarity rather than a phase (`polarity`, `scale`);
  `segments` for a carrier that turns across a long word; and what each match measured published with its frame (`scale`,
  `cfo`, `esn0`), `cfo` taken off the frame when asked, `classify` naming the constellation from the frame's symbols.
  `controls` drive later steps from these fields through a table, as from a header's. A `zc` sequence definition.
- **`{"symbol": "marker"}`** (known symbols before each frame: a sequence with symbols, or a Zadoff-Chu root and shift a
  control sets per frame) and **`{"symbol": "puncture"}`** (symbols at fixed places of every period dropped, and inserted
  on transmit: P25's status symbols).
- **Branches that read part of a frame:** `take` gives a branch item ranges of each frame (an end may be null: to the
  frame's end; `unit` keeps whole groups) and `select` only the frames whose fields (published by a trunk step) an
  expression accepts. Frames are a fixed `frame` length, or start where a trunk control drives the branch step's `frame`
  (frames of any length) (`rpps.profile.branch`).
- **Profiles `p25/tsdu` and `dmr/burst`:** P25 phase 1 control channel TSDUs of one to three TSBKs, each read to the next
  frame sync (NID and TSBKs, CRC-checked) and DMR bursts
  (every slot type; the data bursts' BPTC payloads), with the definitions they use (`constellation/ask/4level`,
  `pulse/rc/0p20_frequency`, `sequence/word/p25_fs` and the eight `dmr_*` sync patterns, link headers `p25_nid`,
  `p25_tsbk`, `dmr_slot_type`, `dmr_bptc`). A framed link header takes a default `length`.
- `examples/zc_modes.py`: a Zadoff-Chu preamble whose root picks each frame's modulation and code.
- **Bursts from real captures** (`rp.analysis.spectrum.burst`, `framing.frames`): the gate takes its levels over 64
  symbols and averages longer as the contrast falls, so it no longer fires on noise (default `contrast` 2.5 on those
  levels); `split` scales its windows and guard times with the occupied bandwidth and looks for guards only 10 dB
  over the floor; each burst's rate is searched within 0.7 to 1.4 times its occupied bandwidth (a long 1100 preamble
  locked it at 2x or 4x); a shared run tolerates single decision errors; long bursts of frames are acquired on the
  unique word they repeat at their own phase (`periodic_word`, `Bursts.period`); slots join within `slot_tol` (now
  16 symbols) by single linkage, and the frame period is learn_plan's or on_off's, whichever the starts' slots fit
  (clustering times occupancy: a divisor of the frame leaves cells empty). The profile's header step tries every bit
  map. A burst acquired on a repeating word takes its carrier from every word (the M-th power's offset over the
  burst, then each word's phase as an anchor) and its timing from the whole burst; a shared run allows one burst in
  four or more to disagree (its length then set against chance for the steps' alphabet); `retime` skips windows over
  known symbols and fits its line by Theil-Sen. The frame period: learn_plan's or on_off's, reduced to the shortest
  divisor the starts cluster at (else their own spacing), then multiplied where the slots fill only some residues of
  the frame index (a divisor of the frame), not where terminals come and go. Each burst's blind timing is the best
  of four quarter-symbol steps, and its carrier the better of the estimate and a quarter of the rate either side (a
  long 1100 preamble is a pair of tones at +-baud/4: it held the timing half a symbol out and the carrier on a line;
  not tried where a quarter turn a symbol is the constellation's symmetry, QPSK), or the middle of such a tone pair
  (`frames.tone_pair`); a burst taken at another carrier is detected again there.
  `Bursts.frame` is None when the starts fall on no slot grid, and `Bursts.plan_frame` is learn_plan's frame: half
  of `frame` where terminals get slots in alternate frames of the plan.
- **Block codes from soft values** (`rp.analysis.code.block.linear.soft_dual`, `soft_find_length`, `soft_score`):
  parity checks from words too noisy for the rank, drawing subsets from the likeliest clean words and keeping checks
  on the tanh-product syndrome. `solve_block(llr=)` uses it when the hard dual comes out short, and finds the length
  from soft values with `soft_length=True`; BCH(63,51) is named blind to 5 dB Eb/N0 (hard bits: above 6 dB).
- **The profile of TDMA bursts notes their units and header** (`rp.analysis.profile`): the code-block unit from the
  bursts' lengths (`burst.units`) and the fields at the start of their payloads (`fields.map`).
- **What each header bit is** (`rp.analysis.framing.fields.map(frames, groups=, lengths=)`): constants, counters (any
  step, either bit order, counted per terminal), terminal ids, length fields, enums and random runs, saved as a link
  header definition (`m.guess.save(name)`: config/header/link/guess). **`rp.analysis.spectrum.burst.units(lengths)`**:
  burst lengths as overhead + n code-block units (MF-TDMA slots of one, two, three blocks), the unit with its error.
- **Bursts with known symbols anywhere, or none** (`rp.analysis.spectrum.burst`): a mid-amble or postamble the bursts
  share (`shared_segments`) is found and each burst acquired on it at its offset (`Bursts.offsets`,
  `BurstResult.known`, `.payload`); bursts that share nothing come back each acquired blind (`Bursts.blind`) instead of
  raising. The gate's smoothing and hang now scale with the occupied bandwidth (`frames.occupied`), so slow bursts are
  no longer cut by their own envelope dips.
- **Bursts that touch are cut apart** (`rp.analysis.framing.frames.split`, run by `gate` unless `touching=False`): at a
  guard time shorter than the gate's hang (a dip longer than the bursts' own envelope dips) or at a power step
  between transmitters, so `burst.demodulate` acquires each one instead of one merged span.
- **TDMA bursts follow their symbol clock and give LLRs** (`rp.analysis.spectrum.burst.demodulate`): a burst of a few
  hundred symbols or more has its clock offset measured (`clock`) and taken out (`burst.retime`), so long bursts no
  longer drift off their timing; each burst's `llr`, and `Bursts.llrs()` for the soft code searches.
- **Blind CRC search** (`rp.analysis.framing.crc.search(frames)`): the generator, init, xorout and where the CRC
  sits (bits it covers, fill after it) from error-free frames as sent, by GCDs of frame differences (RevEng's method on
  bit streams; an uncovered header that varies is cancelled first). Two frame lengths separate init from xorout;
  shipped `config/crc` definitions that fit are named, else `fit.guess.save(name)` writes `config/crc/guess/<name>.json`.
- **A parallel region on a GPU, CUDA or ROCm/HIP alike** (`Parallel("thread", xp=xp)` with `xp, _ =
  rp._accel.device()`): layered min-sum LDPC (`rp.code.kernels.LdpcDecode`), BCH (`BchDecode`: syndromes on the device,
  only words with errors located on the host) and soft demapping (`Demap`) run hand-written device kernels, portable CUDA C
  that NVRTC and hiprtc compile as it is (`rpps.base.kernel.device_functions`); a block interleaver that is a permutation
  gathers (`Permute`), a code a frame switches off passes (`Identity`), and a framed header of fixed size is read in a
  batch (`rp.link.kernels.HeaderDecode`), so the DVB-S2 region stays on the device from one upload to one download. On
  numpy each is the host decoder as before. On an AMD Radeon 860M against 16 CPU threads: the DVB-S2 region 1.9x faster,
  the receiver 1.35x (LDPC 2.1-2.6x, BCH 2.7-3.8x, demap 4.5-7.9x at 16-64 frames a batch); bit for bit with the serial
  runner. Measure with the new `tools/device_bench.py` and `tools/parallel_bench.py --device --batch N`. Building CuPy
  for ROCm and running the GPU tests: Contributing, "Device kernels (GPU)".
- **`timing: mat`, the matched filter and timing in one (`rp.sync.time.MatTrack`)**: a profile fuses its matched filter
  into it, evaluated only at the symbol instants; the timing is the filter's energy maximum (as `Mat`, MatRRC) tracked
  across windows, measured a group of windows at a time so a GPU does one launch a group (`"device": true`). It follows a
  clock 20 ppm off on DVB-S2 like `oerder_meyr`, and any block cut gives the same symbols. With `{"frame": {"device":
  true}}` frame sync's metric runs on the GPU too (`FrameSync(device=True)`); `tools/parallel_bench.py --timing mat
  --device` measures the receiver with both.
- **A region's frame that its header's CRC fails is dropped** (`crc_ok` False) instead of failing the batch.
- **A profile's `requires` narrows a search:** `rp.profile.fits(name, baud=, bw=, Fs=, seconds=, input=)` says why a
  measured signal can't be that profile (None when it may), and `rp.profile.candidates(...)` lists the variants that may.
  `rp.analysis.profile`'s framed search tries only profiles whose rate admits the acquired one, and `identify(baud=)`
  (passed by `AcquireFEC`) names only a profile at that rate. `requires` is checked: an unknown key, an `input` other
  than baseband or audio, or a `from_cf` that isn't a param is a ConfigError.
- **Benchmarks keep their results per machine**: `bench_stages`, `effort_bench`, `parallel_bench`, `resample_bench`
  and `tpc_ber` take `--save`, `--update-baseline`, `--check` (exit 1 on a regression) and `--compare FILE`
  (`tools/_bench.py`), writing to the git-ignored `.benchmarks/<bench>/`. `parallel_bench.py` and `resample_bench.py`
  take their positional arguments through argparse now.
- **`tools/modclass_bench.py`: the Es/N0 at which modulation detection works**, per constellation, for
  `cumulants.classify` (blind and SNR-aided) and `rp.sync.detect` (rate searched or given): the lowest Es/N0 from
  which Pc >= 0.9 holds up to 20 dB over fixed trials, the Pc curve (`--sweep`), or what each modulation and noise
  alone are named with each name's false-discovery rate (`--confusion`, `--frontend` for noise a front-end filter has
  shaped); `--save`/`--check` like the other benchmarks. `tests/unit/test_modclass_threshold.py` keeps the thresholds.
- **SNR before lock, with no noise floor measured apart** (`rp.sync.quality`): `m2m4(y, points)` from the symbols'
  moments under a constellation, `from_sequence(y, ref)` from a known sequence (a CAZAC one: signal and noise from one
  circular correlation, any shift, phase and multipath; else least squares), `cyclic(x, sps)` from the signal's
  spectral correlation at the symbol rate (in band, any pulse: noise outside the signal's band, of any colour, changes
  nothing; the roll-off comes with it), and `blind(x, sps)` from the eigenvalues of the empty band (white noise only:
  None where it can tell it isn't). Measured accuracy in the module.
- **`rp.sequence.zadoff_chu(u, N, shift)`**: Zadoff-Chu (CAZAC) roots; LTE's PSS is made from it.
- **`detect(esn0=, min_evidence=)` and `Lock.find(esn0=, min_evidence=)`**: a known Es/N0 (from a known sequence) to
  score at, and an answer only when sure: `min_evidence` nats of total lead over the runner-up (`Detection.evidence`)
  and 10 over Gaussian noise of the same power (`Detection.over_noise`), else `mod` is None (`best` the leader) and
  `Lock.find` raises NoLock.
- **Catalogue systems and features**: SCPC (IESS-308/309), DVB-RCS/RCS2 return bursts, DVB-S2X (small roll-offs),
  an unmodulated carrier / beacon (`modulation="cw"`), PM TT&C in the C and Ku bands; features `burst` (bursty or
  continuous), `profile` (the profile that decoded it), `sync` (LTE PSS/SSS, NR SSB), `ssb_period`, `subcarrier` and
  `subcarrier_baud`. `python -m rpps ls system` lists them.
- **Definition params take `"choices"`**: the values a param takes, each optionally a bundle of fields one choice sets
  together (`{"param": "modulation.shape"}`, `modulation.bits` in an expression). A value that is not a choice is an error.
  `rp.profile.info` lists them and evaluates `requires` with the chosen bundle.
- **`"when"` on a list item** (a profile's step, any definition's list): kept only where its expression holds.
  Expressions take string literals (`modulation == '16qam'`).
- **Composite checks**: a `config/crc` definition of `"type": "composite"`, CRC parts over transformed views of the
  message (`nibble_swap`, `reverse_bits`, `reverse_bytes`, `invert`, `xor`, `skip`, `take`, or one registered with
  `rp.link.check.register`), concatenated or XORed. It goes wherever a CRC does (an HDLC `fcs`, a link header's `crc`).
- **Blind analysis tries a profile's variants**: `rp.profile.variants(name)` lists the combinations of its params'
  choices (defaults first; a param with `"blind": false` stays at its default), and `identify` (`match_profile`), the
  survey's framed profiles and `rpps solve` try each; a match carries `profile_params` beside `profile_name`.
  `match_profile` returns (name, steps, params).
- **A child profile patches single steps**: `"steps"` as a dict, by step id (`"id"`, else the kind): a dict merges
  into the step, `null` takes it out, `"+before:<id>"` / `"+after:<id>"` / `"+start"` / `"+end"` put steps in
  (`_config.patch_steps`). A list still replaces every step.
- **Checksums**: `config/crc` `"type": "checksum"`, `algorithm` sum, internet (RFC 1071), fletcher (16 / 32), adler
  (Adler-32) or xor (LRC), with word size, endianness, negation, xorout, bit and byte order; it goes wherever a CRC
  does, a composite check's parts too (`rp.link.checksum`).
- **Byte-stuffed framing**: `config/frame` `"type": "bytes"` (flag, escape, escape map or XOR, optional FCS, LSB-first
  bytes), streaming any block cut as HDLC does: `frame/slip`, `frame/kiss`, `frame/ppp_async` (RFC 1662, FCS-16).
- **Fields as definitions** (`rpps.link.fields`): signed, scaled (`scale`, `offset`), per-field `order`, `enum` names,
  `bytes` and `text`, sizes as expressions, `"if"` (per frame, over the fields before), `"from"` positions, groups,
  `"repeat"` (a count, an expression, `"*"`) and `"until"`. Link header `"fields"` take them (a header with conditional
  fields is read as it comes); headers and frames take `"payload_fields"` (in a message's `fields["payload"]`); a
  step's `"message"` fields are a schema. `frame/ax25` reads AX.25 addresses, control and PID (`ax25/afsk1200` uses it);
  `header/ble_adv` names PDU types and reads AdvA and AD structures; `dvbs/qpsk` reads the TS adaptation field and PCR.
- Expressions short-circuit `and` / `or`, as Python does, and take `ceil`, `floor`, `round` and `log2`.
- **Params take `"type"`** (int, float, str, bool), **`"min"` and `"max"`**, checked when a definition loads (and by
  `rp.profile.info`).
- The `mpower` carrier step takes a generated `"shape"` (`"16QAM"`) as the constellation step does.
- `examples/profile_variants.py`.
- **Words of any width**: a `sequence` word (and `word` + `bits` on a step) may be longer than 64 bits; a 100-byte
  unique word is 200 hex digits. `rp.sync.Delimiter` matches a word over 64 bits by FFT correlation.
- **`{"sync": "frame"}` takes a word** (`sequence`, or `word` + `bits`) instead of a `header`: the profile maps it to
  symbols through its own steps, and the frame sync measures its phase, so a unique word sent after the FEC is found
  under any rotation of the constellation (a bit delimiter only corrects an inversion).
- `rp.sync.word_errors(n_bits, false_alarm, positions)`: the most bit errors a delimit step may allow before random
  bits match, and `rp.math.prob.log_binom_half_cdf`.

### Changed
- **The blind puncture search decodes a mother code's patterns together** (`analysis.code.conv.puncture.search`): every
  pattern of a mother decodes on the mother's trellis with erasures where it punctures, so a mother's periods, patterns,
  streams and offsets go through in a batch per length (`puncture.mismatches`, row for row what each punctured code's
  `mismatch_batch` gives): 72 Viterbi calls where there were 1764, the search about 1.5x faster. Its random baselines are
  seeded by the mother, pattern and length, so they no longer depend on what was tried before: the same hypotheses and
  picks, z values within a few tenths of before.
- **`rp.sync.detect` scores each candidate at its own Es/N0 from the symbols' moments** (`quality.m2m4`, the
  quasi-hybrid likelihood test) instead of letting each fit its own noise, which at low SNR named the densest candidate
  (QPSK at 4 dB was 64QAM 19 times in 20, noise alone 32APSK); where the fit's own winner reads over 12.5 dB, the fit,
  as before. It needs only the symbols (no pulse, rate or band assumed) and costs the same. `Detection.how` says
  which was used.
- **`rp.sync.Acquire` decides at fixed stream positions** (min_samples, then doubling, each on the stream's first P
  samples), re-searching modulation and roll-off until two in a row agree, and at the stream's end: its detection no
  longer depends on how the stream is cut into blocks (it decided on whatever had arrived and kept that modulation).
- **`frames.find` sees a short unique word from fewer frames:** its candidate periods sum the correlation over every
  multiple of a lag (`frames.comb_lags`), not the lag alone. A 16-symbol word every 2000 QPSK symbols at 5 dB is found
  from 40 frames (it took 80), and random symbols still give none. A multiple of the period found must be much
  stronger to replace it. `frames.periods` gives `(lag, z)`.
- **`payload.fixed_word` finds a unique word among demodulated bits with bit errors**: the same candidate search, a
  position constant when `agree` (75%) of frames hold its bit, and the word the stretch where they gather, taken when
  chance could not have made it. It gives the word (`word`, `bits`), the bits each frame's copy differs by
  (`mismatches`), and an `errors` to start a delimit step from. A 100-byte word with 10% bit errors is found from 12
  to 20 frames.
- **`matrix_mf` (rpps.sync.time.mat) runs in C++** (`rp._native.matrix_mf`): each row of RRC weights is computed and
  applied in one vectorised pass, never stored, 9-10x the numpy path and equal to it to rounding; `Mat` timing gains the
  most. Both paths now raise IndexError when a symbol's support runs outside x, where numpy's negative indices used to
  wrap round to the end of the block: `Mat`'s +-0.25-sample timing probes use only the symbols whose shifted support
  is in the buffer (`mat.fits`).
- **`signal_profile.framed` reads profiles that end in branches** (`dvbs2/mis`): it read `.stream` off the result,
  which is `{branch: output}` there, and the AttributeError ended the framed pass for every profile. The trunk's
  headers come from the input's stream, each branch's from its own.
- **`AcquireFEC` decides at the end of a stream**: a stream that ends (`flush`) before the window reaches `max_wait`
  is identified on every symbol held past `skip`, blind tools included, where it used to stay "still acquiring" and the
  blind tools never ran. `rp.analysis.profile` sends its last block as the end of the stream, so a short capture gets
  the same look as a long one.
- **`rp.analysis.profile` reads by symbols** (`symbols=2**18`, `max_blocks`): a signal whose code is not decided yet
  and that has fewer than `symbols` symbols keeps reading past `blocks`, up to `max_blocks` (8 x blocks), so a narrow
  signal gets symbols enough for the code scans and the layers inside a code instead of a fixed stretch of the
  capture; a decided one stops costing anything.
- **The layers inside a convolutional code are looked for on nearly all of the window**: `identify.decode_tail`
  skips the first half of the window or the first `settle` (16384) bits, whichever is less, where it skipped the half.
  A DVB-S window of 40k symbols now shows its RS(204,188) behind the Forney interleaver (it took about 64k).
- **`rp.signal.FFTBank`: many channels of a capture from one FFT a hop**, each at its own width and rate (decimated by a
  power of two that keeps 4x its width), at exactly its centre (the bin grid's rounding turned back with a phase ramp
  continuous across hops), zero phase, the same output however the input is cut. `rp.analysis.profile` brings every
  signal to baseband with one bank where each signal's chain shifted and decimated the whole capture itself.
- **`tools/blind_bench.py`: what the blind coding tools break, from how many symbols**: synthetic QPSK streams
  (configured and unconfigured convolutional codes, puncturing, RS behind a Viterbi, scramblers in cascade and with
  three taps, BCH) at several Es/N0, the fewest symbols `identify(blind=True)` names each at, and the seconds it
  takes; `--save`/`--check` like the other benchmarks.
- **Blind identification names an unknown convolutional code and an odd-length block code right**: the convolutional
  scan runs before the block search (a K=6 code was named an LDPC (112,62) code) and, once it finds the generators,
  decodes with them and looks for the layers inside; and the block search keeps the shortest code any bit map shows
  (BCH(63,51) on QPSK came out as a (126,103) code from a map that swaps a symbol's bits).
- **The blind Reed-Solomon search runs in C++** (`rp._native.rs_count_zeros`: the sliding transform's zero counts at
  every phase, bit offset, field and root, the numpy loop the reference), 2.8x on one thread and more on several; it
  was 43% of a 25-signal capture's survey. `identify.scan_rs` loads the configured codes and interleavers once a
  process (they were loaded on every call: 97 s of that survey).
- **`rp.analysis.profile(progress=)`**: called as `progress(stage, done, total, left)` as the work goes (blocks read of
  the stretch planned and the chains still reading, then the profiles built), for a percent and an estimate.
- **A code that carries random-looking bits says so**: when nothing is found inside a convolutional code, `identify`
  keeps `rp.analysis.framing.payload.verdict` of the bits as `fec.payload`, and the profile's payload step reads "the
  bits it carries are random-looking" (keyed or compressed data: no layer to find) instead of only what was not found.
- **`payload.signatures` no longer finds a periodic word in random bits**: a word at a fixed offset (MPEG-TS's 0x47) is
  scored by the binomial deviance and past what the best of the period's columns reaches by chance; 2 hits in 15
  frames where 0.06 were expected read as z 8 (MPEG-TS) and made random payloads "structured".
- **`rp.analysis.framing.randomness`: is a bit stream as random as a cipher's output?** The NIST SP 800-22 rev. 1a
  frequency, block frequency, runs, serial, approximate entropy and cumulative sums tests (each checked against the
  standard's worked examples), a byte chi-square and the bytes' entropy: "random" (encrypted or keyed, or a strong
  compressor's output), "high entropy" (near-uniform bytes that fail some test: compressed, a weak cipher) or "low
  entropy". Entropy alone cannot tell encrypted from compressed data; these tests can for most compressors.
  `payload.verdict` runs them on a random-looking payload (`.randomness`) and says which it is. `rp.math.prob.igamc`,
  the chi-square tail, comes with it.
- **The configured convolutional scan stops trying other rates once one fits overwhelmingly**
  (`scan_defined_conv(decisive=20)`): codes of the same rate are still tried (an equivalent, the generators the other
  way round, must be seen), the rest are not. A K=7 rate 1/2 stream is identified 4x faster.
- **`rp.analysis.profile(jobs=)`** profiles the signals of a capture in that many forked processes (each its share
  of the signals and its own reading of the source from where the profile starts), with the same result: two
  carriers, 31.9 s -> 17.2 s on 2.
- **A profile with a guessed convolutional code no longer fails to build**: `Guess`'s text read an old definition
  layout (`"encode"`); a guess from the generators or from a mother code's puncture now prints, and is a "fec" step.
- **`rp.analysis.profile(hint=)`**: `hint(signal)` gives what an earlier look at the same signal found (its symbol
  rate, the convolutional definitions to try, whether to run the blind tools), so a survey of many captures of the same
  carriers does not search for them again.
- **Equivalent convolutional codes tie exactly in the configured scan**: the error rate a wrong code gets is measured
  once per code shape and shared, where each code drew its own random bits off one generator in turn; which of two
  equivalent codes won (the generators the other way round, an output inverted) depended on the codes tried before.
- **`Mat` keeps a refined timing's first symbol inside its buffer**: moving to a refined detection's lattice could put
  the next symbol's matched-filter support before the samples kept, which now raises (numpy used to wrap round to the
  end of the block); it goes on to the first symbol whose support is there.
- **TDMA bursts in `rp.analysis.profile`**: a bursty signal the continuous chain cannot lock to is demodulated burst by
  burst on its preamble (`spectrum.burst.demodulate`), and the code in each burst kind's payload found where it is laid
  out from the payload's start: `code.block.linear.frame_units(rows)`, the shortest length whose words at 0, n, 2n... of
  every frame are short of rank, on every bit map (one unknown rotation is common to the bursts); then `solve_block`
  names it (BCH, LDPC checks, cyclic...), or the payload's randomness is reported.
- **Survey boxes are fitted to their signals before profiling**: `rp.analysis.profile` reads a stretch of the capture
  once (`source.peek`), takes a fine spectrum of it (the 95th percentile over time), and moves each box's edges to
  where its signal falls 10 dB under its top, or halfway down to the floor around a weak one
  (`signal_profile._refit`), so the channel is cut around the signal, not around the survey's coarse box beside,
  inside or over it. A box holding more than one signal, one whose fit runs into something wider, one under 6 dB over
  the floor, or one already within 8% of the fit is kept; a refitted one says so in a "box" step. On a hand-reviewed
  pool, box width over symbol rate went from 0.98-1.35 to 1.15-1.19 (10-90%), and 3 of 325 boxes reviewers said were
  right moved far.
- **Carriers a notch separates are each profiled**: the survey's spectrum (a few ms, valleys smoothed over a fixed share
  of the band) runs narrow neighbours together into one box. `rp.analysis.profile` now looks at each box in a fine
  spectrum of its own channel samples (the 95th percentile over time, about 400 bins across the box) and, at each
  valley 4 dB under the lower of the peaks either side (more on a noisy spectrum), profiles each part on its own and
  lists every one (`_Chain.carriers`); the box itself is dropped unless it solved as one signal and no part did. On
  reviewed captures this splits 64 of 73 boxes holding several carriers and 6 of 325 holding one.
- **The DVB-S2 PLHEADER reads DVB-S2X MODCODs**: `dvbs2_pl` has a first field `s2x` (EN 302 307-2 5.5.2: b0 = 1 for
  a DVB-S2X MODCOD, coded with one more row and sent a quarter turn on from the SOF); with `s2x` 0 it is DVB-S2's
  header as before. A header definition can turn a field's code words (`"turn": {"field": ..., "degrees": ...}`), and a
  table can apply only where fields have given values (`"only": {"s2x": 0}`, a field not given counting as that value):
  `dvbs2/modcod` has no row for a DVB-S2X header, so its frames are read but not decoded as DVB-S2 ones.
- **`signal_profile.framed` ranks a profile whose payload decoded first**, then by headers: DVB-S2 normal and MIS
  share the PL header, and the one with more headers read could be the one whose payload never decodes.
- **The blind convolutional scan skips generator pairs with a common factor** (`analysis.code.conv.mother.scan_conv`):
  equal generators held on any two equal bit streams, so a BPSK carrier read as QPSK (or fill with I = Q) was found
  as a rate 1/2 code and called solved. Such a pair is a catastrophic code, not one anybody sends.
- **`identify` sees a payload that repeats** (`rpps.analysis.identify.repeat_period`, `CodeDetection.repeat`): fill or a
  test pattern sent again and again made every copy count as evidence, so code tests held on what was only copying.
  A code found on such a stream now stands only if one period shows it too, and `rp.analysis.profile` says the
  symbols repeat (and how often) where no code fits.
- **A framed profile whose frames' payload never decodes is not solved**: `rp.analysis.profile` adds an unsolved
  payload step when none of the profile's last headers (DVB-S2's BBHEADER, a branch point's headers together) decoded,
  where it called a signal solved on its first headers alone. `signal_profile.framed` returns the profile's header ids
  as a sixth item.
- **Framed profiles take the carrier offset from their preamble**: `rp.analysis.profile` ran the framed profiles
  (DVB-S2 and the like) on the chain's symbols with the chain's carrier offset taken off, an M-th power's, which on a
  constellation that isn't M-fold symmetric (16/32APSK) can be off by a good part of the rate: no header decoded, and a
  DVB-S2 APSK carrier stopped as a doubtful 8PSK. The rest of the offset now comes from the preamble's own phase steps
  (`signal_profile.preamble_offset`) before the profile runs, when it stands 3 standard errors clear of the
  preambles' own spread (a noise estimate, applied, drifts a long frame's phase by radians: a 7 dB DVB-S2 carrier's
  BBHEADERs stopped decoding).
- **A block code laid out in each frame is found from few frames**: where the profile finds frames (a unique word) and no
  code fits, the payloads were tried only as one codeword a frame, which needs more frames than the payload has bits.
  Now words laid end to end from each payload's start are looked for too (`linear.frame_units`), and the code is then
  solved on them, a product code's columns included: the fec step is that guess. The payload's start (up to 3 symbols
  back into the unique word, where a payload's first symbols are the same in every frame) and the bit map (the frames'
  common rotation is unknown, and a wrong one makes the code affine, a check short) are the ones where the words are
  shortest of rank. `linear.product` takes several
  columns from each block when there are fewer blocks than the rank needs (`per_block`), and `solve_block` looks for
  columns as long as the bits allow instead of skipping the product test below 128.
- **A survey box that holds carriers side by side is split**: when the cyclic spectrum's rate needs well under the box
  (the chain could not lock to it), the box's own channel samples are surveyed again and each carrier in it profiled on
  its own ("signal 16.1", "signal 16.2"...), its rate acquired alone in its channel; the one that gets furthest stands
  for the box, and the others are listed when they solved or got further than the box did. A piece of the finer survey
  is a carrier only if it is wide enough for the rate and not a lump inside a wider one (or the box itself, found
  again), and a box with more pieces than carriers at the rate could fit is left whole. A box well wider than the rate
  its chain locked to is narrowed to the one carrier in it the same way.
- **Soft decisions in the configured convolutional scan**: `identify` passes the LLRs of each bit map, and a code the
  hard re-encoding test leaves short is scored by the share of |LLR| it disagrees on (`mother.soft_mismatch_batch`,
  decoded from the LLRs), against what random signs on the same reliabilities give. K=7 rate 1/2 on QPSK is identified
  down to Es/N0 2 dB (it was 6).
- **Faster code scans**: `scan_block` screens each word alignment on 24 words before checking all of them (the
  configured block scans 5-6x faster); `scan_defined_conv` screens every code, map and alignment on a fifth of its bits
  (hard, and soft where there are LLRs) before the full Viterbi decode (2.5-4x); `puncture.search` skips a stream that
  is another's bits inverted for mother codes whose generators are of odd weight (an inverted codeword is a codeword).
- **`identify(prior=)`: AcquireFEC's growing window doesn't rescan what it already scanned.** The configured scans look
  at the first 2^15 symbols only, the same in every window from there: a convolutional code found is only decoded
  again for what is inside it, and with nothing configured found the blind pass goes straight to the blind tools (a
  DVB-S stream 4.8 s -> 3.4 s).
- **`AcquireFEC(min_line=5.0)`: no blind tools on a rate Acquire only guessed.** When the rate came from no
  symbol-rate line at least `min_line` over its floor (taken anyway), the symbols are at a wrong rate and no code fits:
  the configured scans still run, the blind ones (the puncture search, block and convolutional searches, the layers
  inside) don't. A rate given (a hint, the cyclic spectrum's second pass) has no line and is always searched.

### Fixed
- A profile's change for a step that had not run yet was placed as if positions started at 0: in a stream that starts
  elsewhere (every burst of a burst profile, at its place in the capture) it landed at the wrong item. An empty position
  map now scales by its step's rate.
- `rp.sync.Burst`'s noise floor stopped at a handful of unusually quiet windows (a narrow channel's windows hold few
  independent samples), and called a whole capture one burst; the floor's cluster must now hold 1% of the windows.
- `BurstTx` takes a burst that starts a little before the last one, within its `lag` (bursts on several carriers at once).
- A block definition's `decoder` that cannot be built from its code (a trellis for a non-linear `expr` code, a code type
  that is not a decoder) is a ConfigError saying why, not an AttributeError.
- A framed link header whose fields are the whole frame (no payload) was restarted before it was read, so no frame came
  out; and a header whose last bit ended a block was read only when more data came, so a stream's last header was lost.
- A profile's output joined frames decoded to bits with frames passed on as LLRs (a code a header switched off) as one
  array, so the bits read as LLRs; such outputs are all LLRs now, a decided bit as sure as the surest soft one.

## 0.6.0 (2026-11-22)

### Breaking
- **RPPS is licensed under the GPL, version 3 or later** (GPL-3.0-or-later), from this release; 0.5.0 and earlier stay
  MIT. A program that distributes RPPS, or a work based on it, does so under the GPL's terms.
- **`stream.dropped` counts source samples**, a radio's overruns and a threaded source's discarded blocks alike (it
  counted blocks for `Blocks.threaded(drop=True)`).
- `ldpc.min_sum(llr)` is `ldpc.propagate(llr)`: the layered schedule runs min-sum or sum-product (the definition's
  `"check"`, or the effort level's). `rpps._effort.iters(base)` is `scale("iters", base)`, `maxlog(base)` is
  `choice("maxlog", base)`.
- **A burst profile's `p.tx()` places the bursts:** a call gives the samples up to its burst's start (the gap before it),
  not the burst; `tx.flush()` gives the rest. Send the packets and write what comes out (`examples/profile_loopback.py`).
  `rpps tx --gap` is seconds, not samples.
- **pysdrlib 0.5 is required.** **Sample formats follow pysdrlib 0.5's names** (SigMF's: the bits of I alone): `cf32` is two float32 (numpy complex64,
  a `.cfile`), `cf64` complex128, `ci16t` int16 pairs (a gqrx `.cs16`), `ci8t` int8 pairs. What RPPS called `cf64` is
  `cf32`, `ci32t` is `ci16t`, `ci16t` is `ci8t`. `rpps rx/tx/survey/profile --fmt` defaults to `cf32`.
  Raw filter files (`import_fir`, `import_iir`, `read_table` `dtype=`) take the same names (`i8t ui8 i16t i32t` for the
  integers) or a numpy dtype with its byte order (`"<i2"`); a bare numpy code is refused (`"i8"` is int64 to numpy).
- `rpps.sync.acquire.refine_timing` is `rp.sync.track_timing(x, sps, beta, span=8, rounds=2)`.
- **The detection rides on each block:** stages read `meta.det` (and `Data.det`), the detection the block went through
  `Acquire` with, not `meta.stream.det`, which is only the latest. A stage of your own that read `meta.stream.det` should
  read `meta.det`; setting `stream.det` by hand still reaches blocks no Acquire stamped.
- **`ble/adv` receives packet by packet:** a burst step cuts the capture into packets and each is demodulated on its own
  (timing and carrier offset from its access address, found by a preamble step). Its runner gives a list, one output per
  packet, each with its own `stream.est`; `sync={"timing": {"method": "known"}}` is no longer needed.
- **Decode status comes with the decode; nothing of it stays on the code.** One loaded code can now decode on several
  threads (before, a thread could read another's status, and get its `failed` mask).
  - Through operators: `out = data / code`, then `out.errors` (per codeword: corrected bits or symbols, -1 where it
    failed). `Coding.errors` is gone.
  - Direct calls: `msg, errors = code.decoder.decode(x, status=True)` replaces reading `decoder.errors` after
    `decode(x)`. The same applies to `fsm.decode_batch(x, status=True)`.
  - CRC: `crc.ok` is gone; `crc.decode(bits, status=True)[1]` is `[0]` if it checked out, `[-1]` if not.
  - LDPC: `converged` is gone; it is `errors >= 0`, and `decode_codewords` returns `(hard, converged)`.
  - HDLC: `last` is gone; `decode(bits, status=True)` gives 0 or -1 per frame found, and `deframe(bits)` every frame
    with its FCS verdict and the aborts.
  - `Marker.errors` (the word's bit errors per frame) is on the decoded data: `out.errors`.
  - `Line.errors` is gone; a Manchester invalid pair is still marked in `out.failed`.
  - Turbo: `decoder.llr` is gone; `decoder.app(x)` gives the a-posteriori LLRs of the message bits (the decoded bits
    are `app(x) < 0`), as the BCJR decoders' `app` does.
  - fsm: the decided paths' output words come from `decoder.paths(x)` (bits, words), not from a private attribute.
  - A code implementation defines `_decode(x, ...) -> (output, errors or None)`; `_code.decode` wraps it.

### Added
- The `mm` timing step takes `biphase` (Manchester symbols, timed on their mid-symbol transition) and `interp`:
  RDS as a profile, built in code and run (`examples/e2e_profile.py`).
- **More of the stable API** (the Building on RPPS guide): `analysis.profile`, `source.Memory`, `waveform.classify`,
  `cumulants.classify`, `fsk.analyze`, `hopping.track`, `burst.demodulate` and `framing.frames` `find`, `cut`,
  `coherence`, `periods`: a change is called out, and a rename warns for a release.
- **`rp.sync.Lock.follow(x, at=pos)`:** the block right after a lock's, its symbols joining on with no symbol dropped
  or repeated at the seam, the carrier phase on the same branch of its M-fold symmetry; so a display gathering symbols
  over contiguous blocks locks only the new block (1.5 ms for 2500 samples, against 21 ms for `update` over 41000).
  `find` and `update` take `at` too; a block that isn't the next raises NoLock, and a timing jump over 0.3 symbol
  makes the lock stale.
- **Channels in shared memory:** `rp.source.ShmSink(name or [names])` publishes blocks (a channelizer's columns, one
  channel each) as pysdrlib shared-memory streams (`pysdrlib.shm.ShmStream`) at their positions, with each channel's
  centre, width, parent span and label for a view. Another process receives one like a radio,
  `rp.source.Device("shm@rpps.<name>")`. Example: `examples/channel_shm.py`.
- **`Device` follows a retune and ends when the device is lost:** a "freq" / "rate" event while streaming cuts the
  block there and starts the stream over at the new setting; pysdrlib's `err.Lost` ends the blocks (the last flushing,
  `device.lost` says why) instead of raising. `Device` also takes a target string (`"hackrf@SERIAL"`, `"shm@rpps.ch3"`,
  `"host/hackrf-0"`), opened with `pysdrlib.open` and closed by `close()`.
- **Load covers everything, and as one number:** `stream.load()` now times pipes (`/`, `*`: decoders) as well as
  stages, and a parallel region's workers; `stream.load(window=s)` over the last `s` seconds of signal;
  `stream.realtime(window=None)` the busiest thread's load (<= 1.0 keeps up).
- **GPU-ready parallel region, vendor neutral:** `rp.parallel.Parallel(..., xp=xp)` runs a region's kernels on an
  array module such as CuPy (CUDA or ROCm/HIP); `rpps._accel.device()` picks it (`RPPS_DEVICE=none|cuda|hip`). A plan
  uploads a batch once, downloads it once, and moves it explicitly around kernels that need numpy (`to:host`,
  `to:device` in `describe()`); kernel constants go to a device once (`Kernel.const`).
- **Positions stay true to time across a gap:** `Device` stamps each block from the radio's own count (pysdrlib's
  `index`), so samples it lost move `meta.pos` on, and every stage carries a gap in its input into its output positions
  (as it does a block `threaded(drop=True)` discarded). `stream.t0` is the radio's clock (`hw_time`) or the host's at
  the first sample, so `meta.time_of(k)` dates a live stream.
- **What an analyser shows of a lock:** `rp.sync.Lock` keeps its block (`lk.x`) and gives `decisions()`, `errors()`
  (per symbol: error vector, EVM and magnitude error as a percentage of the constellation's rms, phase error in degrees),
  `summary()` (EVM rms and peak, MER, magnitude and phase error, and from a fit of the symbols as kI I + kQ Q + c: IQ
  offset, gain imbalance and quadrature error), `derotated()` (x as the symbols see it), `matched(up)` (the matched
  filter between the symbols, `up` points a symbol: eye diagrams; one convolution read between the samples, its
  instants the symbols; `exact=True` the filter's matrix at every point) and `reference()` (the
  decided symbols through the ideal pulse). The measures are `rp.sync.quality.errors` and `summary` for any symbols.
- `Signal.channel()` gives the front end's `delay` [s] and `settle`: the output samples a new stream starts with before
  every filter is full (its start-up transient), so a display drops exactly those.
- **More effort knobs, measured** (`tools/effort_bench.py`: the lowest Es/N0 each chain decodes at, and its speed, per
  level). `max` decodes LDPC by layered sum-product (native `ldpc_sumproduct`: 0.19 dB over min-sum on DVB-S2 64800 1/2
  QPSK, 0.47 dB on 16200 3/5 8PSK, in fewer iterations), lists 4x the polar paths (SC codes list 4; 0.25-0.5 dB), and
  gives a turbo product code one more Chase bit (with 4x its iterations: 0.47 dB). `fast` halves the polar list and takes
  one Chase bit off. An LDPC definition may name `"check": "sumproduct"` itself.
- **Only the message kinds wanted:** `profile.kinds()`, `profile.rx(kinds=...)`, `Decoder(..., kinds=...)`,
  `ProfileInfo.kinds`, `rpps rx --kinds`: only those kinds come out, and only the steps they need run (their steps and the
  headers that drive them): unneeded branches are not built, a chain ends at its last needed step. LTE's MIB alone
  takes about half the time, DVB-S2's PL headers alone skip the LDPC (about 20x faster).
- **Decoded messages:** every frame a profile receives is also an `rp.profile.Message` (kind, fields, payload, crc_ok,
  sample, time, freq, quality), from `runner.take_messages()`, placed in the stream the runner was given. Link headers,
  HDLC, the DVB-S2 PL header, the LTE and NR channels emit them; DVB-S TS packets (with PID) and CCSDS transfer frames are
  cut by the runner (`"message"` on a step); failed frames come too, `crc_ok` False. `rpps rx --messages` prints them as
  JSON lines.
- **What a profile needs:** `"requires"` in every profile (baud, Fs, bw, min_samples, input, from_cf);
  `rp.profile.info(name, baud=, **params)` and `rp.profile.catalogue()` describe them (`ProfileInfo`).
- **`rp.profile.Decoder(name, Fs, offset, cf, baud, bw, **params)`:** a selected signal's samples to the profile's
  messages: shift, staged decimation and a fractional resampler to the profile's rate (an FM demodulator for AX.25's
  audio), the rate measured first where the standard leaves it open. `feed(samples, start=, time=)` takes blocks that need
  not follow on (an overlap is cut, a short gap filled, a long one starts over); `status` says "measuring", "searching",
  "locked" or "lost", why, and the load.
- `analysis.code.block.ldpc`: blind quasi-cyclic codes. `qc_generator` solves the generator's circulants from about
  k/z + 9 words (each word gives z equations; the reliable words solve, held-out ones check), `circulant_size` finds z,
  and `qc_checks` finds sparse checks in the generator itself: sums of two to four shifted parity blocks whose
  information dependences nearly cancel. An LDPC code's dense generator hides those from a word-level search unless the
  words are many.
- `Lock.update(..., recheck=True)` scores the other modulations the search tried on the same symbols and calls the
  lock stale when one fits better (a QPSK lock on 8PSK at 10 dB loses only some 5 dB of Es/N0).
- `rp.analysis.measure_band(x, Fs, freq, bw, cf)`: a band given by hand measured as `find_signals` measures what it
  finds (power, SNR, height, duty), as a `Signal`.
- `rp.wave.demod(..., kind="pm")`: the phase, carrier offset and phase removed, with peak and rms phase deviation.
- The stable API also holds `rrc_pulse`, `spectrum.cyclic` (`alpha_profile`, `features`, `ORDER_NAMES`) and
  `spectrum.scf.coherence`.
- `rp.sync.Lock`: a signal's symbols as a snapshot, for programs that seek, pause and retune. `Lock.find(x, Fs)` searches
  (raises `rp.errors.NoLock` with the reason), `lock.update(block)` follows rate, timing, cfo and phase on any later block,
  consecutive or not, with no search; `lock.stale` says when to search again; `lock.refine(longer)` searches the rate
  again on more signal. `Acquire` runs on it.
- `rp.sync.best_tau` runs the matched filter once and interpolates it (4-20x faster, the same timing); `best_tau`,
  `track_timing`, `matrix_mf` and `centres_in` are exported from `rp.sync`.
- `find_signals(nested=False)` leaves out bands that hold other signals; `Signal.id`, and `Survey.match(previous)` keeps
  ids from survey to survey.
- `rp.wave.demod(x, Fs, kind, bw)`: FM, AM, USB or LSB audio from one block, no state or delay, with FM offset and
  deviation, AM depth, and the tone and SINAD of a test tone.
- README: the API other programs can build on, and an import-cost test (no scipy, matplotlib or pysdrlib on `import rpps`).
- `analysis.spectrum.burst`: bursts of several kinds, each with its own preamble (a TDMA network's burst types each carry
  a unique word): when the bursts share fewer than `min_shared` symbols they are grouped by length, each group's own
  preamble found (`burst_kinds`), and every burst acquired on its kind's (`BurstResult.kind`, `Bursts.preambles`).
- **Speed: LTE and NR downlinks near real time.**
  - `polar_scl`, a C++ polar list decoder: about 20x the numpy one, the same paths and metrics. Polar's N is checked
    without building G.
  - NR's PDCCH decodes each aggregation level's candidates in one batch; the PDSCH's LDPC code is built once per size.
  - NR and LTE cell derotation from a two-table ramp, not an `exp` per sample.
  - LTE CRS channel estimation for every port and symbol in a few array operations; control-region maps cached.
  - Results unchanged: LTE outputs bit-identical on the band 13 capture.
  - NR: 3 s of a LEO pass in 2.3 s at 7.68 MS/s (was 22 s); live 2 s captures at 15.36 MS/s in about 2 s (was 15).
  - LTE: about real time at 15.36 MS/s (was 5x).
  - Cell search: the samples filtered and decimated once, each offset a turn (LTE) or a roll of their spectrum with
    half-size inverse FFTs (NR): LTE 470 ms -> 90 ms, NR +-60 kHz 520 ms -> 250 ms.
- **5G NR downlink to SIB1, through LEO Doppler (NTN):** the profile `nr/downlink` and `rpps.grid.nr` (15 kHz).
  - `NrCell`: PSS search over a wide carrier offset (default +-60 kHz), SSS, the block index and half frame from the PBCH
    DM-RS. Then every SS/PBCH block tracked by two second-order loops: the offset and its rate, the slot timing and the
    clock's rate. Out come whole slots from a frame start.
  - `NrGrid`: a frame per slot. `PBCH` (MIB and timing bits, polar), `PDCCH` (CORESET 0 and the Type0 search space from
    the MIB, CCE-to-REG interleaving, DCI 1_0 to the SI-RNTI), `PDSCH` (DM-RS, TBS, 16 and 64QAM, LDPC, the block's CRC).
  - `nr.tx`: a synthetic downlink, and `leo()`, a satellite pass (delay and Doppler from a circular orbit).
  - Tables `nr/coreset0_15_15`, `nr/type0_fr1`, `nr/pdsch_tdra_a`, `nr/mcs_pdsch_1`, `nr/tbs`. The SS/PBCH block, BCH and
    PBCH scrambling match py3gpp bit for bit (`tests/unit/vectors/nr_py3gpp_ssb.json`).
  - Through a 600 km pass at 2.185 GHz: every MIB and SIB1 over 3 s across closest approach, and at 46 and 50 kHz offsets.
  - On air (HackRF): two live cells at 15 kHz decode to SIB1, one in n5 (MIB 100/100, SIB1 41/42) and one in n25 (MIB
    100/100, SIB1 50/50).
  - The phase NR's upconversion gives each symbol (38.211 5.4): one cell applied it, one (sharing its carrier with LTE)
    did not. Given `fc` (the block's raster frequency), the cell search tells which (its offset measure is not fooled by
    it), and the PDSCH tries both, the CRC choosing.
- **DVB-S2 multiple input streams:** `dvbs2/mis`, one demodulator and decoder, then a branch per input stream. Branches:
  a control whose header is in the trunk drives steps in the branches (the PLS sets every stream's BBFRAME length). Link
  headers: `select` ({field: value}) keeps only the frames with those values (`skipped` counts the others).
  `dvbs2/normal` transmits multistream frames from each frame's `sis` and `isi` fields.
- **5G NR codes:** `nr_polar` (`block/polar/nr_pbch`, `block/polar/nr_dci`): the polar sequence and input interleaver of
  38.212, code size, puncturing and shortening, the sub-block and coded-bit interleavers, CRC-aided list decoding with
  the DCI's 24 ones and RNTI mask (`paths()` for a blind search). `nr_ldpc` (`block/ldpc/nr_bg1`, `nr_bg2`): both base
  graphs with every lifting set, a structured encoder, filler and the punctured 2 Z, the circular buffer at each
  redundancy version and the modulation-order interleaver; `segment()` and `base_graph()` for transport blocks. CRCs
  `crc24_nr_c`, `crc11_nr`, `crc6_nr`; tables `nr/polar_sequence`, `nr/polar_interleave`.
  Checked against py3gpp vectors (`tests/unit/vectors/nr_py3gpp.json`): LDPC and its rate matching, the CRCs and the
  polar encoder are bit-exact.
- **LTE downlink to SIB1:** the profile `lte/downlink` and the package `rpps.grid` (`grid` steps).
  - `grid.lte.LteCell`: cell search (PSS, SSS, carrier offset), then each half-frame's PSS tracked (the sample clock),
    the samples cut into whole half-frames from a subframe 0; acquired again when lost, zeros for what was missed.
  - `grid.lte.LteGrid`: a frame per subframe, its resource grid and the channel of ports 0-3 from the reference signals.
  - `grid.lte.PBCH`, `PCFICH`, `PDCCH`, `PDSCH`: the MIB and port count, the CFI, DCI 1A / 1C in the common search space,
    and the transport block a DCI 1A allocates (turbo, CRC-24A), with transmit diversity for 1, 2 or 4 ports. Each publishes
    what it decoded, so the profile's branches drive one another.
  - `grid.lte.tx`: a downlink (cell signals, MIB, CFI, a DCI and its PDSCH) to test with; the table `lte/tbs`.
  - `rpps rx --branch NAME`: a profile that branches gives that branch's bits.
- **Branches in profiles:** a last `{"branch": ...}` step fans the chain out into branches that read the same blocks
  (`rpps.profile.branch`), in order, each in its own stream; a header in one branch drives steps in the branches after
  it, at the trunk position of its fields; `frame` makes the branches read whole frames. Receive only.
- **Burst transmission in profiles:** `p.tx()` of a profile with a burst step is a `BurstTxRunner`: a call sends one
  burst, through steps built anew for it (no more `flush=True` a packet by hand), and gives the samples up to its start
  (`flush=True` or `tx.flush()`: the rest). The burst step places the bursts (`rp.sync.BurstTx`): `gap_s` apart, or at
  `at=` [s] (TDMA slots), with `ramp_s` raised-cosine power ramps, on a carrier each (`freq=`, or the step's `hop` list
  cycled) on a stream of rate `Fs`: MF-TDMA carriers may share a slot (they add), and a hopping carrier keeps the phase
  of one continuous oscillator. `tx.sent` records each burst's start, length and carrier. A preamble step with
  `"attach": true` sends its own waveform before each burst (`Preamble.attach_tx()`, a `rp.sync.PreambleTx`). `ble/adv` sends 1 ms apart.
  `rpps tx`: `--gap` is in seconds; `--hop`, `--rate-out`. Receiving MF-TDMA or hopping takes a channelizer before the
  profile (a channelized burst step is not built).
- **Burst demodulation in profiles:** a `{"sync": "burst"}` step; the steps after it are built anew for every burst
  (`rpps.profile.burst.BurstRunner`). A `{"sync": "preamble"}` step finds each burst's preamble, its waveform made by the
  profile's own transmit side; on real input (FSK) by the correlation coefficient, with the offset level taken off. Known
  timing takes an `anchor` (a preamble's find). `rpps rx` takes runners that give lists. Live BLE channel 37: every
  packet the hand-written receiver (projects/ble) gets, and 4 more of 106.
- `rp.sync.Burst`: a stream cut into the bursts that rise out of its noise floor, each a `Data` of its own stream at its
  absolute position, so what follows acquires every packet fresh. `Blocks` and `Pipeline` run what follows on each burst
  (a stage may now give a list). Its floor is taken from the noise windows alone (`quiet`), so bursts filling nearly
  all the air are found; zero windows (padding) count as the quietest noise.
- `rp.sync.Preamble`: a burst's preamble by a known waveform (normalized correlation) or its repetition (Schmidl & Cox),
  with the carrier offset the repetition gives, optionally taken off.
- `rp.sync.ToneLock`: an oscillator locked to a pilot, mixing at a harmonic (FM stereo's 38 kHz from 19 kHz; ATSC's pilot
  to DC); windows at absolute positions.
- A source checkout warns on `import rpps` when the in-place `rpps._native` was built from other C++ sources than
  `csrc/rpps_native` has now (`rpps._accel.stale`); rebuild with `python tools/builder.py -c`. Installed packages are not checked.

### Changed
- **`Survey.match` ids are a signal's own:** a signal missing for a survey or a few (`hold`, 8) gets its id back when it
  returns, and a new signal takes an id never used before in the chain, so signals coming and going never renumber the
  others (`survey.lost`, `survey.next_id` carry it on).
- `stream.busy` holds each operator's own time (what it runs inside it is counted for the inner operator, not twice)
  and includes pipes.
- `rp.source.Sink` leaves scaling to pysdrlib's `FileSink(scale=True)` (the inverse of `File`'s) instead of its own:
  the same bytes in every format.
- **Documentation in one place per kind, built with Zensical:** the site (https://anonoei.github.io/RPPS/) has guides
  (`docs/guide/`: the data model, receiving, profiles, definitions, unknown signals, real time, building on RPPS) that
  embed the tested `examples/` scripts, and an API reference rendered from the docstrings by mkdocstrings, replacing
  pdoc. The design that was README.dev.md is in the package docstrings (`rpps.base`: model, rule, categories,
  contracts, type tags, streaming rules; `rpps.sync`, `rpps.profile`, `rpps.parallel`, `rpps.analysis`, `rpps.wave`,
  `rpps.symbol`, `rpps.grid`, `rpps._effort`); its workflow parts are `docs/contributing.md`, the roadmap
  `docs/roadmap.md`, config/README.md `docs/guide/definitions.md`. README.dev.md, `src/rpps/README.md`, the
  subpackage READMEs, `examples/README.md` and `pdoc_templates/` are gone. The `docs` extra is zensical and
  mkdocstrings-python; `python tools/builder.py -d` builds into `site/`. `builder.py` moved to `tools/`.
- **For agents:** AGENTS.md, and `llms.txt` / `llms-full.txt` on the site (AGENTS.md, every
  package docstring and every public signature in one file), generated by `tools/gen_docs.py`.
- `examples/integrating.py`, shown in the Building on RPPS guide: a `Lock` followed across blocks, and a `Decoder` from a
  band at another rate and offset to `Message` records.
- Python blocks in README.md and `docs/` run in the tests (`tests/unit/test_docs.py`), as do checks that every example
  is shown and every package has a docstring.

### Fixed
- **`framing.frames.find` no longer frames noise:** one loud sample or a few positions coherent by chance passed as a
  unique word (65536 random QPSK symbols at 10 dB gave a "period" in 21 searches of 30, a different one each time).
  Products are clipped, and a word is taken only when chance could not have made it over every position and candidate
  period (`false_alarm`, 1e-3); with many frames the per-position threshold comes down, so real frames are found more
  often (13 frames at 10 dB, 0 dB over 65 frames), but never under `min_coherence` (0.2): a structure a few percent of
  frames share (pilot blocks at a short lag) is significant over thousands of frames, and not a unique word.
- **`find_signals` on a crowded band:** a signal a small gap from a louder neighbour was measured against the
  neighbour's level, not the gap's, and dropped out of some surveys; and the receiver's roll-off reaching just past
  the band edge was reported as a narrow signal with a negative SNR.
- `rp.source` refuses a pysdrlib older than 0.5 (ImportError naming the version and the fix): its format names
  differ ("cf32" was two float16), so a capture was read as noise and nothing locked.
- Eye diagrams (`rp.viz.eye`) are upright (they were drawn upside down), take a block with no whole trace (a stream's
  short last block), keep their alpha in range with few traces, and `eye.Q` updates with Q; a debug print is gone.
  `examples/e2e.py` and `e2e_mat.py` draw in rpps' dark style again (`rp.viz.subplots`, `rp.viz.styled()`).
- `analysis.spectrum.waveform.classify` judges the envelope of bursts inside them by the smoothed power
  (`waveform.on_samples`), not by each sample's: filtered PSK in bursts was called constant envelope, its dips between
  symbols dropped.
- `analysis.modulation.cumulants.classify` takes about 2 ms instead of 0.5 s: the moment-cumulant formula merges
  partitions with the same moments once (C84: 109 terms, not 4140), each candidate's theory is computed once, and the
  power share is fitted for every share at once.
- `analysis.spectrum.burst.demodulate` and `framing.frames.bursts`: a given `baud`, `mod` or `beta` narrows each
  burst's acquisition instead of being searched for again (the rate was searched on every burst even when given).
- `rp.sync.Lock.find`, `update` and `refine` raise `NoLock` with a reason for whatever holds no symbols, never numpy's
  errors: samples all zero ("no signal: the samples are all zero"), NaN or inf ("not all finite"), a rate of 0 when any
  line is taken (a constant), estimates that come out non-finite ("nothing like symbols here"). Their arithmetic runs
  under its own numpy error state, so a caller's `np.seterr(all="raise")` no longer turns an underflow into a
  `FloatingPointError`. `update` on zeros now says "all zero" (it said "fewer than 16 symbols").
- `rp.analysis.spectrum.scf.coherence` of silence (all-zero samples) is 0, not NaN with a 0/0 warning (an empty channel
  next to a signal already read 0); samples that are not finite are a `ValueError` saying so.
- `rp.analysis.spectrum.cyclic.features` likewise: silence has no lines whatever `np.seterr` says, samples that are not
  finite are a `ValueError` saying so, and a sequence varying only by rounding (an exactly constant input) no longer
  shows a symbol-rate line found in its residue.
- `rp.sync.BurstTx` (a burst profile's `p.tx()`): a burst added after `take(flush=True)` with an `at` before the
  flushed end failed inside numpy. A flush now ends the stream: the next burst starts a new one, its `at` from that
  stream's start (`sent` is still readable until then); a start in samples already handed out is refused by name.
- `rp.profile.Decoder.feed`: a step back of more than the block (a seek without `reset()`, file playback) was cut as an
  overlap, so every block was dropped until the index passed the old one; it now starts over. A long gap or a seek
  gives the frames the profile held instead of dropping them. `reset()` and a restart forget the block times and the
  last good message (messages took times from the old stream, `status` said "lost" from it).
- **Faster `Lock.update` and `wave.demod`** (a display following a signal every frame):
  - `best_tau` reads each candidate timing's symbol instants off the matched filter with a 12-tap windowed sinc instead
    of upsampling the whole output 16 times by FFT: `track_timing` about 5x faster (8 -> 1.5 ms a block of 8192 samples
    at 20 samples a symbol), and a followed rate no longer biased by the linear read between upsampled points (50007 ->
    50001 Bd on a 50 kBd signal).
  - `sync.candidates` reads each constellation once per config search path: a recheck no longer finds the files every block.
  - `wave.demod` skips the channel filter when `bw` is Fs and the audio filter when `audio_bw` is Fs/2 (the whole span),
    filters the audio on the half spectrum, and keeps complex64 input in single precision: FM over the whole span 20 -> 4 ms
    for 50000 samples.
- `analysis.spectrum.burst`: a long burst type (thousands of symbols, where the gate can open tens of symbols from the
  start) got no preamble of its own and was acquired on another kind's; `burst_kinds` now aligns with a near and a wider
  search. A burst far off carrier (a large share of a turn a symbol) was acquired on data; each burst's blind carrier
  offset now comes off before its preamble search, and the reported `cfo` includes it.
- `analysis.code.block.ldpc.qc_generator` refused words whose information bits include constants (a header's): it now
  solves with those unknowns free and lets the held-out words judge.
- An HDLC frame step flushed with fewer bits held than a flag raised instead of giving nothing.
- `analysis.code.block.linear.sparse_checks` returned nothing when given fewer than k + 16 words (all went to the solve,
  none were left to verify on). It now keeps some aside first.
- Pipelined (`.pipelined`, `.threaded`), a stage after `Acquire` could take a refinement made for a later block, its
  detection or its sps, depending on the threads' timing: the output differed from run to run.
- `sync.time.Mat` dropped a symbol when a refinement moved the timing a little earlier: it now moves to the nearest
  point of the refined lattice, not the first one after.
- `analysis.spectrum.waveform.on_off`: the period of bursts much shorter than their frame came out as the smallest lag
  searched (inside a burst the autocorrelation is near 1 at every lag). It is now the earliest true peak, within 10% of
  the highest, past the autocorrelation's first dip, per lag over its overlap, searched up to half the record (at most
  0.5 s). The power is smoothed over at least 32 samples (a narrow cut's symbols swing at 2 us).

## 0.5.0 (2026-11-20)

### Breaking
- **Loaders take one reference**, the path below the resource: `rp.code.load("conv/k7_171_133/3_4")`,
  `rp.link.load("crc/crc32_mpeg2")`, `rp.signal.filters.load("pulse/rrc/0p35")`. A loaded code is named by that path
  (`"rs/204_188"`), not its family folder.
- **`rp.symbol.load` reads definitions only** (`"psk/qpsk"`); shapes by name come from `rp.symbol.generate("16QAM")`.
  `load_or_generate` is for names a detector hands over.
- **Definition aliases are gone.** Definitions now have parents (`"extends"`, `"abstract"`) and tags
  (`rp.tags()`, `rp.tagged("dvb-s2")`, `rpps ls --tag`):
  - `constellation/dpsk` is gone: DPSK is a constellation after a differential step.
  - `sequence/recurrence/dvb_energy` is `sequence/recurrence/dvbs2_bb`.
  - Convolutional codes have one set of keys for encoder and decoder (the split form is gone).
  - Linear codes take `"n"`, `"k"` (not `"code": [n, k]`).
  - `profile/dvbs` takes the code rate as a parameter; the per-rate profiles pin it.
- **Profile steps name what they load:** `{"code": "conv" | "block" | "interleave", "name": ...}` instead of
  `{"code": "fec", "type": ..., "name": ...}`; a filter step is `{"signal": "filter", "name": "fir/..."}`; a puncture sync
  is `{"sync": "puncture", "conv": ...}`. Your own profiles need the same change.
- **`rpps.analysis` is a tree:** `spectrum/`, `modulation/` (`detect` was `modulation.py`), `sequence/`, `code/conv/`,
  `code/block/` (`linear` was `block.py`), `code/interleave` (was `interleaver.py`), `framing/` (`frames` was
  `framing.py`). No old paths are kept; the top-level helpers (`profile`, `detect`, `find_signals`, `survey`, `psd`) are
  unchanged.
- **Renamed in `rpps.analysis`:**
  - `spectrum.cyclic.spectrum_floor`
  - `sequence.recurrence.bit_windows`
  - `code.conv.tcm.label_windows`
  - `code.interleave.recurrence_z`
  - `spectrum.waveform.on_off`
  - `framing.telemetry.TelemetryFrame`
- **GF(2) helpers live in `rpps.math.gf2`:**
  - `berlekamp_massey` returns `(polynomial, length)`, plus `pdivmod`, `pmul`, `pgcd`, `porder`, `fwht` and `rank_ints`.
  - `code/matrix.py` is gone: use `math.gf2.parity_to_gh` / `gh_to_parity`.
  - `utils.inth.binary.entropy` is `math.bits.entropy`.
- **Removed:**
  - The real-time spectrum viewer (`viz/rt`, `viz/ot`, `viz/blit`, `viz/dialog`) and `analysis/stft` (use `analysis.psd`).
  - `base/buffer`, `source/err`, `convert.read_array`, `errors.DecodeError`.
  - `Signal.baud` (use `width`).
- **Kernels moved:** the kernel protocol (`Form`, `Batch`, `Kernel`) is `rpps.base.kernel`, and `parallel.kernels` is
  gone. Operator control (`publish`, `anchor`, `of`, `Control`) is `rpps.base.control`.
- `rp.viz` no longer sets a global matplotlib style on import.
- `tests/examples` moved to `examples/`, and `tests/dev` and `tests/prop` were removed.

### Added
- **Packet links from definitions:** line codes (NRZI, Manchester, bi-phase), sync markers, header FCS, and streaming
  HDLC. New profiles: BLE advertising, Mode S / ADS-B, AX.25 AFSK1200, CCSDS TM (RS and concatenated).
- **`rpps tx`:** a profile's transmission to a file. `rpps rx` and `rpps show` were also added.
- **Codes:**
  - LDPC with layered normalized min-sum.
  - Turbo product codes (Chase-Pyndiah).
  - Polar (SC, list, CRC-aided).
  - Reed-Muller, Reed-Solomon dual basis, and CCSDS turbo.
- **Sources:**
  - `Source.ppm`, which corrects Fs and cf for the radio's reference error.
  - `Source.REP` / `APPLIED`, and `profile.rx(source=src)`.
  - SigMF and WAV in, WAV out (`rp.source.write_wav`, `WavSink`), and TCP/UDP sources.
- **Signal processing:**
  - `rp.Tee`, which runs one stream through named branches.
  - `rp.signal.baseband`.
  - `sync.CPSync`, OFDM timing from the cyclic prefix.
  - TLL timing.
  - Effort levels (`rp.set_effort`).
- **Blind analysis:**
  - Cumulants, spectral coherence, matrix pencil, FSK / CPM / PM, and impairments.
  - Scramblers and linear recurrences.
  - Turbo constituents and interleavers, polar frozen sets, LDPC reconstruction, TCM checks.
  - Hopping, DSSS, telemetry framing, separation and cancellation, and the catalogue.
- **Faster:** native FIR / IIR / frequency shift, polyphase TX, a thread pool, cached code tables, and blind search in
  about a third of the time.
- `examples/`: one runnable script per recommended way of using RPPS, run by the tests.

### Changed
- `import rpps` no longer imports pysdrlib; File, Device, Sink and the formats import it when used.
- The data path no longer imports `rpps.analysis`.
- The TPC decoder follows Pyndiah: more iterations no longer hurt.

### Fixed
- SigMF sources unmap their data file on close, so Windows can delete it.

## 0.4.1 (2026-11-17)

### Changed
- Wider analysis library, modulations, FEC and scramblers.

## 0.4.0 (2026-11-13)

### Breaking
- Packages renamed: `serial` is `source`; filters, samples and channel are `signal`; `mod` is `symbol`; `coding` is `code`.
- `chain` is gone: a whole signal is a profile (`rp.profile.load`), with header-driven control.
- Sync estimators are block-rate loops, with per-sample loops as the fallback.
- Requires `pysdrlib>=0.4.3`.

### Added
- Optional C++ kernels (`rpps._native`, pybind11) with Python fallbacks.
- The frame-parallel runner, and batch / vectorized kernels.
- FEC definitions for P25, DMR, LTE, ATSC, BLE and Wi-Fi; a CRC of any width and length.
- Remez filter design, a chunked polyphase resampler, Oerder & Meyr estimation, and a least-squares equalizer.
- Signal quality, signal finding, the signal survey and its CLI.
- `projects/`: worked receivers on real signals.
