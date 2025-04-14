import rpps as rp
import numpy as np

def main():
    enc_msg = rp.dobject.StreamData(b"Hello world!")

    mod = rp.mod.load("QPSK")
    mod.set_mapping(mod.get_maps()[0])
    print(mod.constellation)
    mod.constellation.invert()
    print(mod.constellation)
    print(mod)
    # ecc = rp.coding.load("blk", "repeat.3")
    # # scr = rp.scram.load("fdt", "v35")

    # print(f"Mod: {mod}")
    # print(f"ECC: {ecc}")
    # # print(f"SCR: {scr}")

    # m_pipe = lambda msg, ecc=ecc, mod=mod: msg*ecc*mod
    # d_pipe = lambda sym, ecc=ecc, mod=mod: sym/mod/ecc

    # print(f"enc_msg: {enc_msg.hex}")
    # syms = m_pipe(enc_msg)
    # print(f"Got syms {syms}")
    # print()
    # data = d_pipe(syms)
    # dec_msg = rp.dobject.StreamData(data)
    # print(f"dec_msg: {dec_msg.hex}")
    # print(f"Data is the same: {enc_msg.hex == dec_msg.hex}")

if __name__ == "__main__":
    main()
