r"""Analysis: blind estimation and measurement at any representation, outside the data path. The tree follows what is
analysed, the way config/ and the implementation packages do:

 - spectrum: what is in a capture and at what rate. signal (the survey: find_signals, Survey), psd, cyclic (symbol
   rate and carrier lines with a calibrated false alarm, the SCF), scf (every cycle frequency, FRESH), pencil
   (matrix-pencil lines), waveform (tones, OFDM and which system, bursts, envelope), ofdm_pilots (an OFDM grid and its
   pilots), burst (TDMA bursts on a shared preamble), separate (two co-channel signals), hopping (frequency hopping)
 - modulation: what carries the symbols (rate, timing, modulation, offset and phase are rp.sync.detect). cumulants (the constellation's
   family, blind to noise), fsk (FSK / MSK / GMSK), cpm (GMSK and SOQPSK by a per-survivor detector), pm (residual-carrier
   PM and its subcarriers), quality (a gate on hard decisions; Es/N0 is rp.sync.quality), impairments (compression, IQ imbalance,
   phase noise, neighbours), cancel (regenerate and subtract: what hides under a signal)
 - sequence: what was XOR'd in or spread with. recurrence (every recurrence to degree 24, LFSRs, keystreams), scrambler
   (self-synchronising and additive scramblers), spread (DSSS chip rate and code)
 - code: which code. conv (puncture: patterns on a known mother code; rsc: turbo constituents and interleaver; tcm:
   trellis-coded modulation), block (linear: length, alignment and checks by rank, BCH / cyclic, product; ldpc; rs;
   polar), interleave (block and Forney interleavers)
 - framing: how the stream is cut. frames (frame period and unique word on symbols), payload (what decoded bits carry),
   telemetry (PM telemetry end to end), crc (the CRC frames carry: generator, init, xorout, what it covers)
 - code/conv/mother, framing/bits: a convolutional mother code's generators; sync markers, frame length and HDLC on bits
 - identify: which code a hard-bit stream carries, peeling layers (`identify`, which rp.sync.AcquireFEC runs, `solve`)
 - result: what the searches return (Hypothesis, Report, Guess) and `pick`
 - signal_profile: `profile(source)`, every step that processes each signal in a capture, and what is left unsolved
 - catalogue: measured features to named systems (config/system), with the reasons
Results feed the configured stages (`Guess.save` writes a config/fec definition, a found punctured code as a child of
its mother code; rp.sync.Detection fills meta.stream.det; `identify.CodeDetection` names the profile that decodes a stream).

## Which tool answers which question

Tools for signals no definition covers. Each states its statistic and chance level, so "nothing found" is an answer; each
module's docstring has the method, reference and limits. Numpy only, outside the data path (nothing in the data path
imports it, except `AcquireFEC` inside the call). Most take baseband at a few times the signal's width (some noise floor
beside it), locked symbols, or bits. `identify.solve` peels layers off a decoder's output in turn; structure that holds
at every alignment (constant bits, idle words) is not taken for a code. Rank methods need about n + 48 clean words of n
bits; strong codes at low SNR are beyond them on channel bits.

**What is it, and at what rate**

| Question | Tool | How (and limits) |
|---|---|---|
| symbol rate, carrier order and offset | `spectrum.cyclic.features` | lines of `\|x\|^2`, `x^M` and the discriminator against a median floor, calibrated false alarms; rate by harmonic comb |
| every cycle frequency (signals sharing a band) | `spectrum.scf.cycle_frequencies`, `coherence` | FFT-accumulation spectral coherence, conjugate too |
| which constellation | `modulation.cumulants.classify` | C40 ... C84 against each constellation's; orders within QAM / ASK need the SNR |
| FSK / MSK / GMSK, CPM index | `modulation.fsk.analyze`, `cpm_index` | instantaneous frequency; h from the phase lattice |
| GMSK / SOQPSK bits, which pulse | `modulation.cpm.demodulate`, `identify` | per-survivor Viterbi, each survivor its own carrier loop |
| OFDM pilot pattern | `spectrum.ofdm_pilots.discover`, `grid` | grid from CP peaks; period from coherent repetition; pilot lattice, boost, empty cells |
| TDMA bursts, frame and slots | `spectrum.burst.demodulate` | each burst on its shared preamble, data-aided carrier and phase; its symbol clock followed; LLRs (`Bursts.llrs`) |
| two co-channel signals of equal power | `spectrum.separate.estimate`, `separate` | each one's carrier, timing and phase; joint sequence estimate |
| which named system | `catalogue.match`, `features` | measured features against systems from their standards, weighted |
| PM, residual carrier, subcarriers | `modulation.pm.analyze`, `subcarrier`, `nrz` | carrier tracked, phase around it |
| DSSS | `sequence.spread.period`, `chip_rate`, `code`, `despread` | windowed autocorrelation variance; code from eigenvectors |
| frequency hopping | `spectrum.hopping.track`, `dehop` | short-time spectrum clustered into channels and dwells |
| kind of waveform | `spectrum.waveform.classify` | tones, OFDM numerology against known systems, bursts, envelope kurtosis |
| tones in a short record, neighbours | `spectrum.pencil.lines`, `modulation.impairments.neighbours` | matrix pencil; bands split at valleys and level steps |

**Is the lock good, and what is left**

| Question | Tool | How |
|---|---|---|
| can hard decisions be trusted | `modulation.quality.gate` | rotation, drift, near-boundary decisions against the measured Es/N0 |
| how much is locked; what is underneath | `modulation.cancel.cancel`, `align` | regenerate, subtract, residual against the floor |
| amplifier, IQ, phase noise | `modulation.impairments.amplifier`, `iq`, `phase_noise` | ring gains and turns; image ratio; phase error from neighbours |
| separate an interferer | `spectrum.scf.fresh` | least-squares FRESH filter trained on a reference (not blind) |

**Which code, scrambler, interleaver**

| Question | Tool | How |
|---|---|---|
| convolutional code (rate 1/n, K) | `code.conv.mother.scan_conv`, `scan_defined_conv` | generator-pair parity relation against chance; the configured codes by re-encoding |
| puncture of a known mother code | `code.conv.puncture.search` | every rate P/(P+1) pattern through the Viterbi decoder |
| Reed-Solomon over GF(256) | `code.block.rs.search` | shared roots by Galois-field Fourier transform at every byte phase |
| binary block code | `code.block.linear.solve_block` | length from rank deficiency, checks from the null space; then cyclic / BCH, product or LDPC |
| additive scrambler | `sequence.scrambler.additive_search`, `explain`, `search` | Berlekamp-Massey over windows; polynomials factored into known scramblers |
| scrambler taps | `sequence.recurrence.search`, `lfsr`, `keystream` | every recurrence to degree 24 at once; any degree by Berlekamp-Massey |
| trellis code on PSK labels | `code.conv.tcm.search` | Walsh-Hadamard over label windows, every rotation and labelling |
| turbo constituent and interleaver | `code.conv.rsc.constituent`, `turbo_interleaver`, `turbo_interleaver_soft` | 2-bit-label check; permutation from clean or noisy frames |
| LDPC (IRA, QC, large) | `code.block.ldpc.staircase`, `qc_staircase`, `qc_dual`, `qc_size`, `progressive` | one GF(2) solve per structure (a DVB-S2 table from k/360 + 12 clean frames) |
| cyclic / BCH roots | `code.block.ldpc.roots` | zero counts at every beta^j against chance |
| polar frozen bits | `code.block.polar.frozen`, `recognize`, `search` | per-position likelihood over frames (rate 1/2 at -2 dB from 200 frames) |
| interleaver | `code.interleave.block_search`, `forney_search`, `column_delays` | rank deficiency for the size; candidates scored by the code coming back |

**What the bits are**

| Question | Tool | How |
|---|---|---|
| frame period and unique word on symbols | `framing.frames.find`, `cut`, `bursts` | coherence per position over candidate periods |
| sync markers, frame length, HDLC | `framing.bits.find_sync`, `frame_length`, `scan_hdlc` | hits at a period, autocorrelation, flags and FCS against chance |
| PM telemetry end to end | `framing.telemetry.analyze` | BPSK per subcarrier, line code, frame period, counters |
| which CRC, over which bits | `framing.crc.search` | GCD of frame differences (RevEng's method on bit streams); init and xorout from two lengths; config/crc names |
| framing in decoded bits | `framing.payload.signatures` | E1/T1, ATM, GFP, HDLC, IPv4, CCSDS ASM, MPEG-TS, fixed bits, each against chance |
| structured, scrambled or random | `framing.payload.verdict` | signatures, bias, linear complexity, compressibility |
| encrypted, or only compressed | `framing.randomness.test` | NIST SP 800-22 frequency, block, runs, serial, approximate entropy, cusum; byte chi-square and entropy |
"""
from . import spectrum, modulation, sequence, code, framing
from . import result, identify
from . import signal_profile
from . import catalogue
from .signal_profile import profile, SignalProfile
from .spectrum.signal import Signal, Survey, Channel, find_signals, measure_band, survey, welch, noise_floor
from .spectrum.psd import psd, vbw, vbw_calc
