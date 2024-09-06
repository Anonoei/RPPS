from pyboiler.logger import Logger, Level

import time
import cProfile
import pstats

import numpy as np
import rpps as rp

def main():
    mod = rp.mod.load("QPSK")
    mod.set_mapping(mod.get_maps()[0])
    # ecc = rp.coding.name("BLK", "Repetition", 3)

    enc_msg = rp.dobject.StreamData()
    enc_msg.stream.bytes = b"""Hello World!"""

    f_pipe = lambda inp:inp @ mod
    r_pipe = lambda inp:mod @ inp

    syms = f_pipe(enc_msg)

    print(syms)
    print(type(syms))

    data = r_pipe(syms)
    print(data)
    print(data.stream)
    print(rp.base.Stream.from_bytes(data.stream.bitarray.to_bytes()))

if __name__ == "__main__":
    # pr = cProfile.Profile()
    # pr.enable()
    main()
    # pr.disable()
    # ps = pstats.Stats(pr).sort_stats(pstats.SortKey.CUMULATIVE)
    # ps.print_stats()
