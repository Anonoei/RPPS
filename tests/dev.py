import time
import cProfile
import pstats

import rpps as rp

def main():
    mod = rp.mod.load("QPSK")
    mod.set_mapping(mod.get_maps()[0])
    ecc = rp.coding.name("BLK", "Repetition", 3)

    enc_msg = rp.dobject.StreamData(b"Hello World!")

    print(f"enc_msg: {enc_msg.hex}")
    # f_pipe = lambda inp:inp @ ecc @ mod
    # r_pipe = lambda inp:inp @ mod @ ecc

    print(f"Running {ecc}")
    coded = enc_msg @ ecc
    print(f"coded: {coded}")
    print(f"Running {mod}")
    syms = coded @ mod
    print(f"syms: {syms}")

    print(f"Running {mod}")
    demod = syms @ mod
    print(f"demod: {demod}")
    print(f"Running {ecc}")
    decode = demod @ ecc
    print(f"decode: {decode}")

    dec_msg = rp.dobject.StreamData(decode)

    print(f"dec_msg: {dec_msg.hex}")
    print(f"{enc_msg.hex} == {dec_msg.hex} : {enc_msg.hex == dec_msg.hex}")

if __name__ == "__main__":
    # pr = cProfile.Profile()
    # pr.enable()
    main()
    # pr.disable()
    # ps = pstats.Stats(pr).sort_stats(pstats.SortKey.CUMULATIVE)
    # ps.print_stats()
