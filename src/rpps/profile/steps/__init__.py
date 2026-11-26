"""What each step of a profile builds (rpps.profile)

A step is one dict with one category key, `{"<category>": "<kind>", ...its parameters}`, in receive order. Building it
gives a Step: the operators it runs on receive and on transmit, and the representations it takes and gives on receive.

    transform steps (wave, symbol, code, link) build a transmit/receive pair: the receive side as written, the transmit
        side inverted (the profile runs transmit steps in reverse order)
    sync and signal steps run on receive only

A step names the resources it uses by their path below the resource ("additive/dvb_energy", "rrc/0p35"), so the profile
says what each is.

Sync steps name what is estimated; how is a method, chosen by `defaults` in the profile or the `sync` argument of
`rp.profile.load`, else the default here.

Every step has an id ("id", else its kind), by which a profile's `controls` name it, and says which of its parameters a
header may change and how (`controls`: {parameter: "live" | "restart"}, from its operators' CONTROLS). A parameter it
does not list cannot be controlled (see rpps.profile.control).

The kinds (`python -m rpps steps` lists them with the sync methods):

    signal   filter (a definition), lowpass / highpass / bandpass / bandstop (designed in Hz: cutoff + taps, or transition +
             atten_db), decimate, resample (up/down), channelize (freq, width), shift (freq), conjugate (spectral inversion),
             dc_block, deemphasis
    every    class: your own operator by module path, in the category it does ("class", "args"; wave: "tx_class")
    sync     timing, carrier, equalize, agc (methods), frame (a physical header's preamble, a word or a complex
             sequence, or named alternatives / a Zadoff-Chu set; before, length, segments, scale, polarity, classify), delimit (a word among bits:
             word + bits, a word sequence, or a link header's preamble; length, keep, polarity, errors), align (a repeating
             marker), puncture, pilots, squelch, quality (Es/N0 and EVM to est), ofdm, inversion (by a header's preamble
             or a complex sequence)
    wave     shape (a pulse definition; symbol pulses get a matched filter), stagger, fm (h + sps: FSK; deviation: analog),
             am (the envelope; alpha null keeps its mean), ssb (bw, sideband), ofdm (n_fft, n_cp)
    symbol   constellation (name, mapping), differential, spread (a chip sequence), carriers, header, pilots, scramble,
             marker (known symbols before each frame: a `sequence` with symbols, or `zc` {length, root, shift}; root,
             shift, sequence controlled per frame), puncture (symbols at `at` of every `every` from the frame start,
             dropped; `value` inserted on transmit: P25's status symbols)
    code     fec (type block / conv / interleave, name, args), scramble (name, args; length: frames of that many bits from
             the first), line (invert, nrzi, nrzm, manchester, thomas), marker (a word before each frame: length, or frames by
             control), uncoded
    link     crc, frame (HDLC), header (fields, crc, fcs, payload; select: the frames with those field values), gse
             (GSE packets of DVB-S2 data fields: protocols, each PDU a frame)
    grid     resource grids (rpps.grid), receive only: lte_cell (offset, span, resolution, acquire, lose, gate, pci, duplex), lte_grid
             (n_rb), and the channels on its frames, lte_pbch, lte_pcfich, lte_pdcch (rntis), lte_pdsch; nr_cell (offset,
             span, resolution, period, acquire, lose, gate, fc, scs, pattern), nr_grid, nr_pbch, nr_pdcch (rntis,
             min_bw), nr_pdsch (fc, min_bw)
"""
from ._core import (STAGE, DECODE, ENCODE, Step, REGISTRY, METHODS, register_step, register_sync_method, kinds, methods,
                    context, build)
from . import _signal, _sync, _grid, _transform # (each registers its kinds on import)
from ._sync import frame_words
from ._transform import constellation
