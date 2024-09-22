import rpps as rp

def main():
    enc_msg = rp.dobject.StreamData(b"Hello world!")

    # mod = rp.mod.load("QPSK")
    # mod.set_mapping(mod.get_maps()[0])
    ecc = rp.coding.load("blk", "repeat.3")
    # scr = rp.scram.load("fdt", "v35")

    # print(f"Mod: {mod}")
    print(f"ECC: {ecc}")
    # print(f"SCR: {scr}")

    enc_msg = rp.dobject.StreamData(b"Hello World!")
    print(f"enc_msg: {enc_msg.hex}")
    data = enc_msg * ecc
    print(data)
    data = data / ecc
    dec_msg = rp.dobject.StreamData(data)
    print(f"dec_msg: {dec_msg.hex}")
    print(f"Data is the same: {enc_msg.hex == dec_msg.hex}")

    # syms = enc_msg * scr * ecc * mod

    # print()

    # data = syms / mod / ecc / scr

    # dec_msg = rp.dobject.StreamData(data)
    # print(f"dec_msg: {dec_msg.hex}")
    # print(f"Data is the same: {enc_msg.hex == dec_msg.hex}")

if __name__ == "__main__":
    main()
