"""BER / FER of the turbo product code (64,57)^2 (`block/product/tpc_64_57_sq`, Chase-Pyndiah) over BPSK / AWGN, per iteration count

    python tools/tpc_ber.py                          # chase 4 and 5, 1 / 2 / 4 / 8 iterations, 2.5 .. 3.5 dB
    python tools/tpc_ber.py --words 400 --chase 5 --iters 8 --ebn0 2.5 2.75 3.0

Reference: AFF3CT error_rate_references, TURBO_PROD/AWGN/BPSK/BCH1/Turbo_prod_BCH_64_57_p5_i8.txt (extended BCH (64,57)
squared, chase 5, 8 iterations, alpha 0.5 on every half iteration): BER 1.2e-2 / 1.2e-3 / 1.2e-5 at 2.5 / 2.75 / 3.0 dB.

Findings (2026-11-20, 200 words a point, chase 5, 8 iterations):
    decoder                                  2.5 dB           2.75 dB          3.0 dB          (BER / FER)
    alpha 0.2 .. 1.0, beta 0.2 .. 1.0        1.3e-1 / 0.95    6.5e-2 / 0.51    4.3e-3 / 0.045
    alpha 0, 0.2, 0.3, 0.5 (held), beta      1.8e-2 / 0.65    1.9e-3 / 0.12    0    / 0
    AFF3CT                                   1.2e-2 / 0.50    1.2e-3 / 0.07    1.2e-5 / 0.0013
so about 0.1 dB from the reference. Before the fix, a bit with no competing candidate got the extrinsic
beta*mean|input|*D - input, which pushes against the channel on reliable bits, and more iterations did worse: chase 4 at 3.5 dB
went from 5e-4 (2 iterations) to 2.3e-2 (4). Pyndiah's alpha rising to 1.0 also lets words that fail diverge when iterated
past his 4 iterations (BER at 8 above BER at 4); holding it at 0.5, as AFF3CT does, keeps BER falling with iterations.
"""
import argparse
import time

import numpy as np

import rpps as rp

REF = {2.5: (1.20e-2, 0.50), 2.75: (1.20e-3, 7.03e-2), 3.0: (1.19e-5, 1.29e-3), 3.25: (2.93e-7, 7.45e-5)} # AFF3CT, chase 5, 8 iterations


def run(words, chase, iters, ebn0, seed=1):
    tpc = rp.code.load("block/product/tpc_64_57_sq")
    dec = tpc.decoder
    dec.chase = chase
    k, n = tpc.encoder.num, tpc.encoder.den
    for eb in ebn0:
        rng = np.random.default_rng(seed)
        msg = rng.integers(0, 2, words*k).astype(bool)
        x = tpc.encoder.encode(msg)
        sigma = np.sqrt(n/k/(2*10**(eb/10)))
        llr = 2*((1 - 2.0*x) + rng.normal(0, sigma, len(x)))/sigma**2
        cols = []
        for it in iters:
            dec.iters = it
            t = time.time()
            e = (dec.decode(llr) != msg).reshape(words, -1)
            cols.append(f"{it}: {e.mean():.1e}/{e.any(axis=1).mean():.3f} ({time.time() - t:.1f} s)")
        ref = f"   AFF3CT {REF[eb][0]:.1e}/{REF[eb][1]:.3f}" if eb in REF and chase == 5 else ""
        print(f"chase {chase}  {eb:5.2f} dB  raw {np.mean((llr < 0) != x):.3f}   " + "   ".join(cols) + ref, flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--words", type=int, default=100)
    ap.add_argument("--chase", type=int, nargs="+", default=[4, 5])
    ap.add_argument("--iters", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--ebn0", type=float, nargs="+", default=[2.5, 2.75, 3.0, 3.25, 3.5])
    a = ap.parse_args(argv)
    print("BER/FER per iteration count")
    for c in a.chase:
        run(a.words, c, a.iters, a.ebn0)


if __name__ == "__main__":
    main()
