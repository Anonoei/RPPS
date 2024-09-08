import time
import cProfile
import pstats

import rpps as rp


def main():
    mod = rp.mod.load("QPSK")
    mod.set_mapping(mod.get_maps()[0])
    ecc = rp.coding.name("BLK", "Repetition", 3)
    scr = rp.scram.load("fdt", "fib")

    print(f"Mod: {mod}")
    print(f"ECC: {ecc}")
    print(f"SCR: {scr}")

    enc_msg = rp.dobject.StreamData(b"Hello World!")
    syms = enc_msg * scr @ ecc @ mod  # Encode data with ecc, and mod. Get the symbols

    data = syms @ mod @ ecc / scr  # Read the symbols

    dec_msg = rp.dobject.StreamData(data)
    print(f"{enc_msg.hex == dec_msg.hex}")  # Check decoded data is what you encoded


if __name__ == "__main__":
    # pr = cProfile.Profile()
    # pr.enable()
    main()
    # pr.disable()
    # ps = pstats.Stats(pr).sort_stats(pstats.SortKey.CUMULATIVE)
    # ps.print_stats()
