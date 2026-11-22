# Changelog

RPPS is 0.x: a minor version may rename or remove things, and old names are not kept as aliases. Every release lists
what breaks under **Breaking**, with what to use instead. Work in progress (not yet here) is in [WORKING.md](WORKING.md).

## Unreleased

### Breaking
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
  `csrc/rpps_native` has now (`rpps._accel.stale`); rebuild with `python builder.py -c`. Installed packages are not checked.

### Fixed
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
