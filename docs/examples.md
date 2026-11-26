# Examples

Each script in [examples/](https://github.com/Anonoei/RPPS/tree/main/examples) shows the way RPPS recommends doing one
thing, and checks that it worked. Run them from the repository root (`python examples/<name>.py`). The ones in the first
table make their own signals and need nothing else; `tests/unit/policy/test_examples.py` runs them, and the guides show them in
full.

| Script | Shows | In |
| --- | --- | --- |
| quickstart.py | Bits to symbols and back in ten lines | [The data model](guide/data-model.md#bits-to-symbols-and-back) |
| operators.py | `*` encodes, `/` decodes, links made of operators, partial blocks, `Uncoded` before a link layer | [The data model](guide/data-model.md#operators-links-and-partial-blocks) |
| soft_decisions.py | LLRs from the demodulator straight into the decoder, and what deciding early costs | [The data model](guide/data-model.md#soft-decisions) |
| channel_and_sync.py | A simulated channel built from stages, and blind recovery with the block estimators | [Receiving](guide/receiving.md#a-simulated-channel-and-recovery-with-block-estimators) |
| streaming_io.py | Sources and sinks (SigMF), iterating vs by hand (`stream=`, `ctx=`), pipelined threads and their load | [Receiving](guide/receiving.md#sources-sinks-and-carried-state) |
| profile_loopback.py | A known signal from one definition: a profile transmits with header fields and receives in any block cut | [Profiles](guide/profiles.md#transmit-and-receive) |
| parallel_rx.py | A profile's frame-parallel region on threads and processes, the same output as the serial runner | [Profiles](guide/profiles.md#more-than-one-core) |
| own_definitions.py | A new signal as JSON definitions in your own config directory, one profile extending another | [Definitions](guide/definitions.md#your-own-signal-in-your-own-directory) |
| profile_variants.py | One profile, several variants (choices and `when`), and an HDLC FCS built from two CRC-8s (a composite check) | [Definitions](guide/definitions.md#variants-choices-and-when) |
| zc_modes.py | A Zadoff-Chu preamble whose root picks each frame's modulation and code, through a table, frame by frame | [Profiles](guide/profiles.md#a-preamble-that-says-what-follows) |
| mftdma.py | An MF-TDMA network: the burst time plan and clock read from the forward link (DVB-RCS2 tables in GSE), each return burst decoded and named by its slot and terminal; then a plan learned from the bursts | [Profiles](guide/profiles.md#burst-time-plans) |
| list_definitions.py | Every shipped definition, as `python -m rpps ls` prints them | [Definitions](guide/definitions.md) |
| integrating.py | `rp.sync.Lock` found once and followed on later blocks; `rp.profile.Decoder` from a band at another rate and offset to `Message` records | [Building on RPPS](guide/integrating.md#a-lock-and-a-profiles-messages) |
| unknown_signal.py | `rp.analysis`: survey a capture, then name the profile for each signal or say what is unsolved | [Unknown signals](guide/unknown-signals.md) |

These need more than the package:

| Script | Shows | Needs |
| --- | --- | --- |
| [e2e.py](https://github.com/Anonoei/RPPS/blob/main/examples/e2e.py) | FM RDS by block estimation (Oerder & Meyr timing, M-th power carrier), live plots | the [PySDR RDS recording](https://pysdr.org/content/rds.html) at `data/fm_rds_250k_1Msamples.iq`, matplotlib |
| [e2e_mat.py](https://github.com/Anonoei/RPPS/blob/main/examples/e2e_mat.py) | The same RDS receiver with block estimators and no loops | the same recording, matplotlib |
| [e2e_profile.py](https://github.com/Anonoei/RPPS/blob/main/examples/e2e_profile.py) | The same RDS receiver as a profile built in code and run; RDS blocks found by their checkwords, the station's name read | the same recording |
| [live_profile.py](https://github.com/Anonoei/RPPS/blob/main/examples/live_profile.py) | Survey and profile the signals of a live radio | a radio or I/Q server (pysdrlib) |
| [window.py](https://github.com/Anonoei/RPPS/blob/main/examples/window.py) | The window functions and their spectra | matplotlib |

Worked receivers on real signals (FM with RDS, BLE, ADS-B, Wi-Fi, ATSC, LTE, P25/DMR, spectrum sweeps, TDOA) are in
[projects/](https://github.com/Anonoei/RPPS/tree/main/projects), each with its own README.
