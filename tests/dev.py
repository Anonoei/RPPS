import rpps as rp
import numpy as np
import matplotlib.pyplot as plt

def main():
    enc_msg = rp.Data("Hello world, again and again!")
    # enc_msg = rp.dobject.StreamData()

    mod = rp.mod.load("QPSK")
    mod.set_mapping(mod.get_maps()[0])
    ecc = rp.coding.load("blk", "repeat.5")
    scr = rp.scram.load("fdt", "v35")

    print(f"Mod: {mod}")
    print(f"ECC: {ecc}")
    print(f"SCR: {scr}")

    print(f"enc_msg: {enc_msg.hex}")

    print(mod.constellation)
    syms = enc_msg*ecc*scr*mod

    upsample = rp.sample.up.Zeros(4)
    pulse = rp.filters.ShapingRRC(101, 4, 0.35)

    syms.data = pulse.run(upsample.run(syms.data))

    rp.viz.ot.time.IQ3d(syms.data)
    # rp.viz.phasor(None, syms.data)
    plt.show()

    mod.constellation.invert()

    print(mod.constellation)
    data = syms/mod/scr/ecc
    dec_msg = rp.Data(data)

    print(f"dec_msg: {dec_msg.hex}")
    print(f"Data is the same: {enc_msg.hex == dec_msg.hex}")

if __name__ == "__main__":
    main()
