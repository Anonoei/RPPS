# Changelog

RPPS is 0.x: a minor version may rename or remove things, and old names are not kept as aliases (except the stable API
in the Building on RPPS guide, which warns for one release). Every release lists what breaks under **Breaking**, with
what to use instead. Work in progress (not yet here) is in [WORKING.md](WORKING.md).

## Unreleased

## 0.6.0 (2026-11-22)

### Breaking
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
- **What an analyser shows of a lock:** `rp.sync.Lock` keeps its block (`lk.x`) and gives `decisions()`, `errors()`
  (per symbol: error vector, EVM and magnitude error as a percentage of the constellation's rms, phase error in degrees),
  `summary()` (EVM rms and peak, MER, magnitude and phase error, and from a fit of the symbols as kI I + kQ Q + c: IQ
  offset, gain imbalance and quadrature error), `derotated()` (x as the symbols see it), `matched(up)` (the matched
  filter between the symbols, `up` points a symbol: eye diagrams; its instants are the symbols) and `reference()` (the
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
