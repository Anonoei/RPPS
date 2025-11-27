import rpps as rp
import numpy as np
import matplotlib.pyplot as plt

# From https://pysdr.org/content/rds.html

def main():
    num_samps = 1024*8
    sink = rp.serial.File("cf64", "data/fm_rds_250k_1Msamples.iq")
    meta = rp.Meta(Fs=250e3, cf=99.5e6)

    mod = rp.mod.FSK()

    fig, ax = plt.subplots(2)

    for samps in sink(num_samps):
        fig.suptitle(f"{sink.percent*100:.2f}%, {sink.cur_samp}/{sink.max_samp}")
        meta.obj = samps
        rp.viz.freq.psd(ax[0], meta.obj, meta.Fs, vbw_hz=1000)
        meta.obj = mod.decode(samps)
        rp.viz.freq.psd(ax[1], meta.obj, meta.Fs, vbw_hz=1000)
        plt.show()
        exit()
        # ax[0].cla()
        # ax[1].cla()
        meta.reset()

if __name__ == "__main__":
    main()
