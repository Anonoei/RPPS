from pyboiler.logger import Logger, Level

import time

import numpy as np
import rpps as rp


def main():
    # path = rp.viz.get_file()
    # print(f"Path is: {path}")
    path = "data/QPSK_2SPS.sigmf-data"
    meta = rp.Meta()
    meta.fmt = "cf32"
    # meta.freq["SampleRate"] = 250000
    meta.freq["CenterFreq"] = 96900000

    # ax = rp.viz.figure().subplots()

    # rp.viz.ion()
    syms = []

    for sym in rp.file.read(meta, path, 64, offset=4898816):
        cur_time = rp.file.get_format().cur_time
        read_time = rp.file.get_format().read_time
        # rp.viz.cla()
        # rp.viz.psd(sym, meta, ax)
        # ax.set_title(f"PSD ({cur_time:.2f}/{read_time:.2f})")
        # rp.viz.pause(0.01)
        syms.append(sym)
        # percent = rp.file.get_format().block / rp.file.get_format().blocks
        print(rp.progress.bar("dev_viz", cur_time, read_time, "Processing..."), end="\r")
    print(rp.progress.bar("dev_viz", read_time, read_time))
    print(f"Sample rate is {meta.freq['SampleRate']}")

    # rp.viz.ioff()

    rp.viz.psd(syms[0], meta)
    rp.viz.spectrogram(syms, meta, rp.file.get_format())
    rp.viz.show()


if __name__ == "__main__":
    main()
