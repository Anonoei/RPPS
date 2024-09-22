import rpps as rp

def main():
    enc_msg = rp.dobject.StreamData(b"Hello world!")

    mod = rp.mod.load("QPSK")
    mod.set_mapping(mod.get_maps()[0])
    ecc = rp.coding.load("blk", "hamming.7_4")
    scr = rp.scram.load("fdt", "v35")

    print(f"Mod: {mod}")
    print(f"ECC: {ecc}")
    print(f"SCR: {scr}")
    print()

    enc_msg = rp.dobject.StreamData(b"Hello World!")
    print(f"enc_msg: {enc_msg.hex}")
    syms = enc_msg * scr
    syms = syms + ecc
    syms = syms @ mod
    data = syms @ mod
    data = data - ecc
    data = data / scr

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
