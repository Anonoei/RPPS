import time
import cProfile
import pstats

import rpps as rp


def main():
    mod = rp.mod.load("QPSK")
    mod.set_mapping(mod.get_maps()[0])
    ecc = rp.coding.name("BLK", "Repetition", 3)
    scr = rp.scram.load("fdt", "galois")

    print(f"Mod: {mod}")
    print(f"ECC: {ecc}")
    print(f"SCR: {scr}")
    print()

    enc_msg = rp.dobject.StreamData(b"Hello World!")
    print(f"enc_msg: {enc_msg.hex}")
    syms = enc_msg * scr @ ecc @ mod

    data = syms @ mod @ ecc / scr

    dec_msg = rp.dobject.StreamData(data)
    print(f"dec_msg: {dec_msg.hex}")
    print(f"Data is the same: {enc_msg.hex == dec_msg.hex}")

if __name__ == "__main__":
    # pr = cProfile.Profile()
    # pr.enable()
    main()
    # pr.disable()
    # ps = pstats.Stats(pr).sort_stats(pstats.SortKey.CUMULATIVE)
    # ps.print_stats()
