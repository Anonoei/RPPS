import rpps as rp
import numpy as np
import matplotlib.pyplot as plt

def main():
    enc_msg = rp.dobject.StreamData(b"Hello world!")

    mod = rp.mod.load("QPSK")
    mod.set_mapping(mod.get_maps()[0])

    mod.draw_refs()

    mod.constellation.invert()
    mod.draw_refs()
    plt.show()

if __name__ == "__main__":
    main()
