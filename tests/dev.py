import rpps as rp
import numpy as np
import matplotlib.pyplot as plt

def main():
    msg = "Hello world, again and again!"
    print(f"Encoding {msg}")
    enc_msg = rp.Data(msg)

    mod = rp.mod.load("QPSK")
    mod.set_mapping(mod.get_maps()[0])
    ecc = rp.coding.load("blk", "repeat.5")
    scr = rp.scram.load("fdt", "v35")

    print(f"Mod: {mod}")
    print(f"ECC: {ecc}")
    print(f"SCR: {scr}")

    # print(type(ecc))
    # exit()

    # print(f"enc_msg: {enc_msg.hex}")

    # print(mod.constellation)
    syms = rp.Data(enc_msg.data)
    syms = syms*ecc*scr*mod
    print(f"Encoded to {syms}")

    # shift = rp.filters.ShiftFreq(25)

    # sps = 8
    # pulse = rp.filters.ShapingMatRRC(12, sps, beta=0.35)
    # shaped = pulse.burst(syms.data)

    # syms.data = shift.run(syms.data, 250_000)
    # rp.viz.time.IQ(None, pulse.pulse)
    # plt.show()
    # exit()

    # rp.viz.ot.time.IQ3d(syms.data)
    # fig, ax = plt.subplots()
    # t = np.arange(0, 256)*sps
    # t_p = np.arange(0, 256*sps)
    # ax.plot(t, syms.data[:256].real, ".-")
    # ax.plot(t_p, shaped[:256*sps].real, ".-")
    # rp.viz.time.I(ax, syms.data[:256])
    # rp.viz.phasor(None, syms.data)
    # plt.show()

    # mod.constellation.invert()

    dec_msg = syms/mod/scr/ecc

    dec_msg = dec_msg.as_bytes()
    enc_msg = enc_msg.as_bytes()

    print(f"dec_msg: {dec_msg.hex}")
    print(f"Data is the same: {enc_msg.hex == dec_msg.hex}")
    print(f"Decoded: {dec_msg.data.tobytes().decode('utf-8')}")

if __name__ == "__main__":
    main()
