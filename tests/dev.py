import rpps as rp
import numpy as np
import matplotlib.pyplot as plt

def main():
    msg = "Hello world, again and again!"
    # print(f"Encoding {msg}")
    enc_msg = rp.Data(msg)

    mod = rp.symbol.load("QPSK")
    mod.set_mapping(mod.maps[0])
    # ecc = rp.code.load("block", "repeat/5")
    ecc = rp.code.load("block", "hamming/7_4")
    scr = rp.code.scram.load("v35")

    # print(f"Mod: {mod}")
    # print(f"ECC: {ecc}")
    # print(f"SCR: {scr}")

    # print(enc_msg(scr+ecc+mod))

    syms = enc_msg * scr * ecc * mod
    # print(f"Encoded to {syms}")

    dec_msg = syms / mod / ecc / scr

    dec_msg = dec_msg.as_bytes()
    enc_msg = enc_msg.as_bytes()

    print(f"dec_msg: {dec_msg.hex}")
    print(f"Data is the same: {enc_msg.hex == dec_msg.hex}")
    print(f"Decoded: {dec_msg.data.tobytes().decode('utf-8')}")

if __name__ == "__main__":
    main()
