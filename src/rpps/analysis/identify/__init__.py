"""Blind FEC identification

Takes hard bits (what a demodulator produced, with an unknown bit mapping, rotation and alignment)
and asks which code they are most likely carrying. Every test is a parity relation that holds on
codewords but is a coin flip on anything else, so a hypothesis is scored by how far its relation
sits from 50% (a deviance z-score), then corrected for how many hypotheses were tried.

 conv: rate 1/n feed-forward convolutional codes. For generators g_a, g_b: (c_a * g_b) xor (c_b * g_a) = 0
 block: any linear block code / LDPC given its parity-check matrix H, at every word alignment
 scramblers: an additive scrambler of period T just adds a fixed pattern to the relation, so folding the relation
    by T (with `periods`) finds it. Self-synchronising scramblers are undone with `descramblers` first.

The tests are about structure, not about being right: a noisy channel weakens them, and the verdict
is None when nothing stands out.
"""
from ._streams import CONV_SYMBOLS, REPEAT_AGREE, diff_alphabets, bit_streams, repeat_period
from ._scans import (SCREEN_WORDS, FIRST_CHECKS, RS_WORDS, repeat_check, scan_block, config_blocks, whiteness, detect_fec,
                     decode_tail, scan_rs, guess_block)
from ._profiles import match_profile
from ._detection import CodeDetection, _named
from ._blind import solve, _distinct
from ._core import identify
