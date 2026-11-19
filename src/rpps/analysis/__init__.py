"""Analysis: blind estimation and measurement at any representation, outside the data path

 - fec: which error-correcting code (convolutional, block/LDPC, RS, scrambler) a hard-bit stream carries
 - signal_profile: `profile(source)`, every step that processes each signal in a capture, and what it might be where unsolved
 - rs, scrambler, block, puncture: blind Reed-Solomon codes, self-synchronising scramblers, binary block codes (BCH / cyclic,
   product, LDPC: length, alignment and checks by rank) and puncture patterns, for what no definition covers
 - waveform: what a signal is when it isn't one linearly modulated carrier: tones, OFDM (and which system), bursts,
   constant envelope
 - framing: a symbol stream's frame period and unique word, frames cut and turned on it (blind)
 - cyclic: symbol rate and carrier lines with a calibrated false-alarm probability (a "no line" verdict too), the
   spectral correlation function
 - cumulants: higher-order cumulants (C40 C42 C63 ...) of any constellation, blind to noise: the modulation's family
 - fsk: FSK / MSK / GMSK from the instantaneous frequency
 - pm: residual-carrier phase modulation: the carrier, index, phase, rates and subcarriers in it
 - telemetry: PM telemetry end to end: each subcarrier's BPSK, line code (Manchester, differential) and frames
 - tcm, recurrence, interleaver, ldpc: trellis-coded modulation checks, every scrambler recurrence to degree 24,
   interleaver size and shape, LDPC reconstruction (blind, beside rs / scrambler / block)
 - rsc, polar: turbo constituent codes and interleaver (from noisy frames too), polar frozen positions and construction
   (blind; successive cancellation at low SNR)
 - cpm: GMSK and SOQPSK demodulated by a per-survivor sequence detector, blind to rate, carrier and timing
 - ofdm_pilots: an OFDM signal's time-frequency grid and its pilot pattern (period, lattice, continual, boost)
 - burst: TDMA bursts demodulated one by one on their shared preamble; the frame and slots
 - separate: two co-channel signals of equal power, each one's carrier and timing, both symbol streams
 - catalogue: measured features to named systems, with the reasons
 - payload: what decoded bits carry (framing signatures, or a randomness verdict)
 - impairments: amplifier compression (AM/AM, AM/PM), IQ imbalance, phase noise, neighbouring signals
 - scf, pencil: every cycle frequency (co-channel signals, FRESH filtering); matrix-pencil lines in short records
 - cancel: regenerate a locked signal and subtract it: the excess over the floor, and what hides under it
 - modulation: symbol rate, timing, modulation, frequency offset and phase from baseband samples
 - quality: Es/N0, Eb/N0 and EVM of locked symbols, and a gate on hard decisions before blind tests
 - psd: spectra; rate: symbol rate; signal: finding the signals in a capture (survey)
Results feed the configured stages (`Guess.save` writes a config/fec definition, `Detection` fills meta.stream.det,
`fec.CodeDetection` (from `fec.identify`, which rp.sync.AcquireFEC runs) names the profile that decodes a stream).
"""
from . import fec
from . import rs, scrambler, block, puncture # Blind tools for what no definition covers (see fec.solve)
from . import signal_profile
from .signal_profile import profile, SignalProfile
from . import modulation
from . import waveform
from . import framing
from . import quality
from . import cyclic, cumulants, fsk, pm, telemetry
from . import tcm, recurrence, interleaver, ldpc, payload, cancel
from . import rsc, polar, impairments, scf, pencil
from . import cpm, ofdm_pilots, burst, separate, catalogue
from .modulation import detect, Detection

from . import rate
from . import signal
from .rate import find_rate, find_baud
from .signal import Signal, Survey, Channel, find_signals, survey, welch, noise_floor
from .psd import psd, vbw, vbw_calc
