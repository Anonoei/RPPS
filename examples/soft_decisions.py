"""Soft decisions: keep the demodulator's LLRs all the way into the decoder

    python examples/soft_decisions.py

`symbols / constellation` gives LLRs, not bits: log(P0/P1), positive for a 0, so a hard decision is `llr < 0`. Every
decoder that can use them does (Viterbi, BCJR, LDPC, turbo, polar, Hamming on its trellis, repetition); one that can't
(Reed-Solomon) decides itself. So the recommended chain is just `noisy / mod / code`, with no hard decision in between.
This script measures what deciding early costs: the same K=7 rate 1/2 convolutional code, decoded from LLRs and from
hard bits, over QPSK in white noise.
"""
import numpy as np

import rpps as rp

def main(esn0s=(1.0, 2.0, 3.0, 4.0), nbytes=4000):
    rng = np.random.default_rng(0)
    qpsk = rp.symbol.load("psk/qpsk")
    conv = rp.code.load("conv/k7_171_133/1_2")
    data = rng.integers(0, 256, nbytes).astype(np.uint8)
    sent = np.unpackbits(data).astype(bool)

    def ber(out):
        got = out.as_bits().data
        return np.mean(got != sent[:len(got)])

    print(f"{conv}, QPSK, {8*nbytes} bits a point")
    print(" Es/N0   channel    hard Viterbi  soft Viterbi")
    rows = []
    for esn0 in esn0s:
        syms = rp.Data(data, flush=True) * conv * qpsk
        noisy = syms + rp.signal.channel.AWGN(esn0_db=esn0, seed=7) # unit-power symbols: Es/N0 is the SNR per symbol
        llr = noisy / qpsk # Mod (CHANNEL/LLR)

        # Hard: decide first. `as_hard()` (like as_bits / as_bytes) converts the Data it is called on in place, so copy
        # it when the soft values are still needed
        hard = llr.copy().as_hard()
        raw = np.mean(hard.data != (rp.Data(data, flush=True) * conv).data[:len(hard)])
        h, s = ber(hard / conv), ber(llr / conv)
        rows.append((esn0, h, s))
        print(f"{esn0:5.1f} dB  {raw:9.2e}  {h:12.2e}  {s:12.2e}")

    # Soft is worth about 2 dB here: it must never be worse, and clearly better where the channel is noisy
    assert all(s <= h for _, h, s in rows)
    assert rows[1][2] < rows[1][1]/5, "soft decisions should be far better than hard at 2 dB"

if __name__ == "__main__":
    main()
