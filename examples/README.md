# Examples

Each script shows the way RPPS recommends doing one thing, and checks that it worked. Run them from the repository
root (`python examples/<name>.py`); start with quickstart.py. The ones in the first table make their own signals and need nothing else;
`tests/unit/test_examples.py` runs them.

| Script | Shows |
| --- | --- |
| [quickstart.py](quickstart.py) | Bits to symbols and back in ten lines: the README's first example |
| [list_definitions.py](list_definitions.py) | Every shipped definition, as `python -m rpps ls` prints them |
| [operators.py](operators.py) | The data model: `*` encodes, `/` decodes, links made of operators, partial blocks, `Uncoded` before a link layer |
| [soft_decisions.py](soft_decisions.py) | LLRs from the demodulator straight into the decoder, and what deciding early costs (hard vs soft Viterbi) |
| [channel_and_sync.py](channel_and_sync.py) | A simulated channel built from stages, and blind recovery with the block estimators (`Acquire`, `Mat`, `FFCarrier`) |
| [profile_loopback.py](profile_loopback.py) | A known signal from one definition: a profile transmits with header fields and receives in any block cut |
| [own_definitions.py](own_definitions.py) | A new signal as JSON definitions in your own config directory, one profile extending another |
| [streaming_io.py](streaming_io.py) | Sources and sinks (SigMF), iterating vs by hand (`stream=`, `ctx=`), pipelined threads and their load |
| [parallel_rx.py](parallel_rx.py) | A profile's frame-parallel region on threads and processes, the same output as the serial runner |
| [unknown_signal.py](unknown_signal.py) | `rp.analysis`: survey a capture, then name the profile for each signal or say what is unsolved |

These need more than the package:

| Script | Shows | Needs |
| --- | --- | --- |
| [e2e.py](e2e.py) | FM RDS with per-sample loops (M&M, Costas), live plots | the [PySDR RDS recording](https://pysdr.org/content/rds.html) at `data/fm_rds_250k_1Msamples.iq`, matplotlib |
| [e2e_mat.py](e2e_mat.py) | The same RDS receiver with block estimators and no loops | the [PySDR RDS recording](https://pysdr.org/content/rds.html) at `data/fm_rds_250k_1Msamples.iq`, matplotlib |
| [live_profile.py](live_profile.py) | Survey and profile the signals of a live radio | a radio or I/Q server (pysdrlib) |
| [window.py](window.py) | The window functions and their spectra | matplotlib |
