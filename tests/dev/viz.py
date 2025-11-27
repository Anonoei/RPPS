import rpps as rp
import numpy as np
import matplotlib.pyplot as plt

from rpps.viz.plots import freq

def main():
    # num_samps = 1024*8
    num_samps = 1024*2
    Fs = 250_000
    sink = rp.serial.File("cf64","data/fm_rds_250k_1Msamples.iq")
    samps = sink.read(num_samps)

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    # lines = ax
    # exit()
    # writer = rp.viz.save.FFmpegMP4(fig, fps, "video_time3d.mp4")
    # writer.open()

    for samps in sink(num_samps):
        # samps *= rp.filters.window.blackman(len(samps))
        # lines = rp.viz.freq.psd_minmax(lines, samps, Fs, 1000)
        samps *= 10
        ax = rp.viz.ot.time.IQ3d(samps, Fs, ax)
        break
        # rp.viz.pause(0.01)
        ax.cla()

    # writer.close()

if __name__ == "__main__":
    main()
