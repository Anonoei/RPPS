import rpps as rp
import numpy as np

def main():
    # enc_msg = rp.dobject.StreamData(b"Hello world!")

    # mod = rp.mod.load("QPSK")
    # mod.set_mapping(mod.get_maps()[0])
    ecc = rp.coding.load("cnv", "cnv.1_2_3")
    # scr = rp.scram.load("fdt", "v35")

    # print(f"Mod: {mod}")
    print(f"ECC: {ecc}")
    # print(f"SCR: {scr}")

    # enc_msg = rp.dobject.StreamData(b"Hello World!")
    enc_msg = rp.dobject.BitObject(np.array([0,0,1,1,0], dtype=bool))
    print(f"encoding {enc_msg}")
    # print(f"enc_msg: {enc_msg.hex}")
    data = enc_msg * ecc
    print(data)
    data = data / ecc
    print(f"decoded {data}")
    #dec_msg = rp.dobject.StreamData(data)
    #print(f"dec_msg: {dec_msg.hex}")
    #print(f"Data is the same: {enc_msg.hex == dec_msg.hex}")

    # syms = enc_msg * scr * ecc * mod

    # print()

    # data = syms / mod / ecc / scr

    # dec_msg = rp.dobject.StreamData(data)
    # print(f"dec_msg: {dec_msg.hex}")
    # print(f"Data is the same: {enc_msg.hex == dec_msg.hex}")

if __name__ == "__main__":
    main()
