"""Analysis: blind estimation and measurement at any representation, outside the data path. The tree follows what is
analysed, the way config/ and the implementation packages do:

 - spectrum: what is in a capture and at what rate. signal (the survey: find_signals, Survey), psd, rate, cyclic (symbol
   rate and carrier lines with a calibrated false alarm, the SCF), scf (every cycle frequency, FRESH), pencil
   (matrix-pencil lines), waveform (tones, OFDM and which system, bursts, envelope), ofdm_pilots (an OFDM grid and its
   pilots), burst (TDMA bursts on a shared preamble), separate (two co-channel signals), hopping (frequency hopping)
 - modulation: what carries the symbols. detect (rate, timing, modulation, offset, phase), cumulants (the constellation's
   family, blind to noise), fsk (FSK / MSK / GMSK), cpm (GMSK and SOQPSK by a per-survivor detector), pm (residual-carrier
   PM and its subcarriers), quality (Es/N0, EVM, a gate on hard decisions), impairments (compression, IQ imbalance,
   phase noise, neighbours), cancel (regenerate and subtract: what hides under a signal)
 - sequence: what was XOR'd in or spread with. recurrence (every recurrence to degree 24, LFSRs, keystreams), scrambler
   (self-synchronising and additive scramblers), spread (DSSS chip rate and code)
 - code: which code. conv (puncture: patterns on a known mother code; rsc: turbo constituents and interleaver; tcm:
   trellis-coded modulation), block (linear: length, alignment and checks by rank, BCH / cyclic, product; ldpc; rs;
   polar), interleave (block and Forney interleavers)
 - framing: how the stream is cut. frames (frame period and unique word on symbols), payload (what decoded bits carry),
   telemetry (PM telemetry end to end)
 - code/conv/mother, framing/bits: a convolutional mother code's generators; sync markers, frame length and HDLC on bits
 - identify: which code a hard-bit stream carries, peeling layers (`identify`, which rp.sync.AcquireFEC runs, `solve`)
 - result: what the searches return (Hypothesis, Report, Guess) and `pick`
 - signal_profile: `profile(source)`, every step that processes each signal in a capture, and what is left unsolved
 - catalogue: measured features to named systems, with the reasons
Results feed the configured stages (`Guess.save` writes a config/fec definition, a found punctured code as a child of
its mother code; `Detection` fills meta.stream.det; `identify.CodeDetection` names the profile that decodes a stream).
"""
from . import spectrum, modulation, sequence, code, framing
from . import result, identify
from . import signal_profile
from . import catalogue
from .signal_profile import profile, SignalProfile
from .modulation.detect import detect, Detection
from .spectrum.rate import find_rate, find_baud
from .spectrum.signal import Signal, Survey, Channel, find_signals, survey, welch, noise_floor
from .spectrum.psd import psd, vbw, vbw_calc
