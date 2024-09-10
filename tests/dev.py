import time
import cProfile
import pstats

import rpps as rp
import numpy as np

def main():
    enc_msg = rp.dobject.StreamData(b"Hello World!")
    msg_bit = rp.dobject.ensure_bit(enc_msg)
    print(f"enc_msg: {enc_msg.hex}")
    input_size = 3
    past = np.array([1,1,1,0], dtype=bool)
    polys = np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0],[1,1,1,1]], dtype=bool)
    conv = rp.coding.conv.Convolutional(input_size, past, polys)
    coded = np.array([], dtype=bool)
    for i in range(0, len(msg_bit), 3):
        c = conv.code(msg_bit.data[i : i + 3])
        coded = np.append(coded, c)
    print(f"MSG bits: {len(msg_bit)} {msg_bit.data.astype(int)}")
    print(f"CNV bits: {len(coded)} {coded.astype(int)}")

    exit()
    mod = rp.mod.load("QPSK")
    mod.set_mapping(mod.get_maps()[0])
    ecc = rp.coding.name("BLK", "Repetition", 3)
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
