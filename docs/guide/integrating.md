# Building on RPPS

These are kept stable: a change to what they take or give is called out in the CHANGELOG, and a name is not moved
without the old place going through a release that warns. Anything else (a module's other functions, `rpps._*`, a
stage's state) can change.

| Where | What |
| --- | --- |
| `rp.Data`, `rp.Stage`, `rp.Pipe`, `rp.Pipeline` | the data and the operators (`*`, `/`, `+`) |
| `rp.sync.Lock`, `rp.errors.NoLock` | a signal's symbols as a snapshot: `Lock.find(x, Fs)`, `lock.update(block)` (`recheck`: other modulations), `lock.follow(next_block, at=pos)` (contiguous blocks, their symbols joined with no slip), `lock.stale`, `lock.refine(longer)`; NoLock (with its reason) for anything that holds no symbols |
| `lock.decisions()`, `errors()`, `summary()`, `derotated()`, `matched(up)`, `reference()` | what an analyser shows of the lock's block: each symbol's error, the error summary (EVM, MER, IQ offset, gain imbalance, quadrature error), the matched filter between the symbols (eye), the ideal waveform |
| `rp.sync.detect`, `rp.sync.Detection`, `rp.sync.Acquire` | acquisition once, and as a stream stage |
| `rp.sync.best_tau`, `rp.sync.track_timing`, `rp.sync.matrix_mf`, `rp.sync.centres_in`, `rp.sync.find_baud` | timing and rate |
| `rp.sync.quality.measure`, `errors`, `summary` | Es/N0, Eb/N0, EVM of symbols; each one's error; an analyser's error summary |
| `rp.analysis.find_signals`, `rp.analysis.survey`, `Survey`, `Signal` | the signals in a capture; `Survey.match` keeps their ids |
| `rp.analysis.measure_band` | a band given by hand, measured as a found one (a `Signal`) |
| `Signal.channel()` | the filters that bring a signal to baseband: `front`, `Fs`, `decimation`, `delay`, `settle` (its start-up transient, in output samples) |
| `rp.analysis.profile`, `rp.source.Memory` | every step that processes a signal, and what it might be where unknown, over a capture held in memory (`Memory.capture(device, n)`) |
| `rp.analysis.spectrum.waveform.classify`, `rp.analysis.modulation.cumulants.classify`, `rp.analysis.modulation.fsk.analyze` | what kind of waveform (tones, on-off, OFDM, single carrier), which constellation, FSK's tones and deviation |
| `rp.analysis.spectrum.hopping.track`, `rp.analysis.burst.demodulate` | a hopping signal's dwells; TDMA bursts, each demodulated from its preamble (`rp.analysis.spectrum.burst.demodulate` until the next release, warning) |
| `rp.analysis.framing.frames`: `find`, `cut`, `coherence`, `periods` | frame structure in symbols: the period, the frames cut at it, how alike they are |
| `rp.analysis.spectrum.cyclic`: `alpha_profile`, `features`, `ORDER_NAMES`; `rp.analysis.spectrum.scf.coherence` | cyclic features: symbol rate and modulation order lines, spectral coherence |
| `rp.signal.filters.rrc_pulse` | the root raised cosine pulse at any (fractional) time |
| `rp.wave.demod`, `rp.wave.FMDemod`, `AMDemod`, `SSBDemod` | analog audio (FM, PM, AM, SSB), one block or a stream |
| `rp.profile.ls`, `rp.profile.load`, `rp.profile.info`, `rp.profile.catalogue` | the shipped profiles, and what each needs (`requires`) |
| `rp.profile.Decoder`, `rp.profile.Message`, `runner.take_messages()` | a selected signal's frames, decoded, one record each |
| `profile.kinds()`, `profile.rx(kinds=...)`, `Decoder(..., kinds=...)` | only some message kinds, and only the steps they need |
| `rp.source.formats` | sample formats, by pysdrlib's names |

## A lock, and a profile's messages

A display following one signal holds an `rp.sync.Lock`; a monitor decoding a band it picked holds an
`rp.profile.Decoder`, fed blocks with where they start, and reads `Message` records back:

```python
--8<-- "examples/integrating.py"
```

## Import cost, logging, definitions

`import rpps` loads numpy and RPPS only (about 0.2 s): no scipy, matplotlib or pysdrlib until something asks for them
(`rp.viz`, a source), so programs can import it at start-up (`tests/unit/test_import_cost.py`).

Logging is off until you ask: `rp.logger.enable("DEBUG")`. Definitions of your own load without touching the package:
`rp.add_config_path("my_config")` or `RPPS_CONFIG` ([Definitions](definitions.md)).
