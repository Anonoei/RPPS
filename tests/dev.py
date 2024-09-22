import time
import cProfile
import pstats

import rpps as rp
import numpy as np

def main():
    enc_msg = rp.dobject.StreamData(b"Hello world!")
    msg_bit = rp.dobject.ensure_bit(enc_msg)


    #ecc = rp.coding.load("blk", "hamming.7_4")
    ecc = rp.coding.generate("linear")
    p = np.array([
        [1,1,0],
        [1,0,1],
        [0,1,1],
        [1,1,1]
    ])
    ecc = ecc(7, 4, p)
    exit()

    print()
    print(f"ECC: {ecc}")
    print()

    print(msg_bit)

    enc_dat = msg_bit + ecc
    enc_dat.data[128] = not enc_dat.data[128]

    print(enc_dat)
    dec_msg = enc_dat - ecc
    dec_msg = rp.dobject.StreamData(dec_msg)

    print(f"enc_msg: {enc_msg.hex}")
    print(f"dec_msg: {dec_msg.hex}")
    print(f"Data is the same: {enc_msg.hex == dec_msg.hex}")

    exit()
    mod = rp.mod.load("QPSK")
    mod.set_mapping(mod.get_maps()[0])
    # ecc = rp.coding.name("BLK", "Repetition", 3)
    scr = rp.scram.load("fdt", "v35")

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
