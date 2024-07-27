from pyboiler.logger import Logger, Level

import time
import cProfile
import pstats

import numpy as np
import rpps as rp

def main():

    mod = rp.mod.identify.by_name("QPSK", 0)
    ecc = rp.coding.Repetition(2)

    pipeline = rp.Pipeline(mod, ecc)

    enc_msg = b"""Test"""

    def encode(pipeline: rp.Pipeline, enc_msg):
        time_start = time.perf_counter()
        syms = pipeline.enc(enc_msg)

        time_dur = time.perf_counter() - time_start
        print(f"Encode took {time_dur:.2f}s")

        #rp.viz.DrawConstellation(syms, meta)
        #mod.draw_refs()
        rp.viz.show()

        return pipeline.meta.serialize(syms)

    def decode(pipeline: rp.Pipeline, path):
        time_start = time.perf_counter()
        data = pipeline.from_file(path)

        time_dur = time.perf_counter() - time_start
        print(f"Decode took {time_dur:.2f}s")

    path = encode(pipeline, enc_msg)
    print()
    decode(pipeline, path)

if __name__ == "__main__":
    # pr = cProfile.Profile()
    # pr.enable()
    main()
    # pr.disable()
    # ps = pstats.Stats(pr).sort_stats(pstats.SortKey.CUMULATIVE)
    # ps.print_stats()
