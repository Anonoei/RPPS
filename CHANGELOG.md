# Changelog

RPPS is 0.x: a minor version may rename or remove things, and old names are not kept as aliases. Every release lists
what breaks under **Breaking**, with what to use instead.

## Unreleased

### Breaking
- **`ble/adv` receives packet by packet:** a burst step cuts the capture into packets and each is demodulated on its own
  (timing and carrier offset from its access address, found by a preamble step). Its runner gives a list, one output per
  packet, each with its own `stream.est`; `sync={"timing": {"method": "known"}}` is no longer needed. Packets closer
  together than 80% of the time (a continuous stream) are not found: the burst floor needs quiet.
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
- **Burst demodulation in profiles:** a `{"sync": "burst"}` step; the steps after it are built anew for every burst
  (`rpps.profile.burst.BurstRunner`). A `{"sync": "preamble"}` step finds each burst's preamble, its waveform made by the
  profile's own transmit side; on real input (FSK) by the correlation coefficient, with the offset level taken off. Known
  timing takes an `anchor` (a preamble's find). `rpps rx` takes runners that give lists. Live BLE channel 37: every
  packet the hand-written receiver (projects/ble) gets, and 4 more of 106.
- `rp.sync.Burst`: a stream cut into the bursts that rise out of its noise floor, each a `Data` of its own stream at its
  absolute position, so what follows acquires every packet fresh. `Blocks` and `Pipeline` run what follows on each burst
  (a stage may now give a list).
- `rp.sync.Preamble`: a burst's preamble by a known waveform (normalized correlation) or its repetition (Schmidl & Cox),
  with the carrier offset the repetition gives, optionally taken off.
- `rp.sync.ToneLock`: an oscillator locked to a pilot, mixing at a harmonic (FM stereo's 38 kHz from 19 kHz; ATSC's pilot
  to DC); windows at absolute positions.
- A source checkout warns on `import rpps` when the in-place `rpps._native` was built from other C++ sources than
  `csrc/rpps_native` has now (`rpps._accel.stale`); rebuild with `python builder.py -c`. Installed packages are not checked.

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
