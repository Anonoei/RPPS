from pyboiler.logger import Logger, Level

import time
import numpy as np

import rpps as rp


def main():
    r_file = "rpps.r.bin"
    w_file = "rpps.w.bin"

    meta = rp.Meta()
    meta.fmt = "cf32"

    mod = rp.mod.name("QPSK", 0)
    ecc = rp.coding.name("BLK", "Repetition", 3)

    pipeline = rp.Pipeline(mod, ecc)

    inter = rp.inter.File(meta, pipeline, 1000, r_file, w_file)

    inter.start()

    while True:
        write_data = input("Enter data: ").encode()
        inter.write(pipeline.enc(rp.Stream.from_bytes(write_data)))


if __name__ == "__main__":
    main()
