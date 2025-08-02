import rpps as rp
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as manim
import time

def main():
    sink = rp.serial.File("f32","/home/ano/sync/dev/anonoei/RPPS/data/fm_rds_250k_1Msamples.iq")

    samp_time = 10 # ms
    samp_show = 100 # ms
    Fs = 250_000
    Ts = 1/Fs
    num_samps = int((samp_time/1000)*Fs)
    fps = (1000/samp_show)

    nfft = 1024
    overlap = 0.8
    window = rp.filters.window.blackman
    hist_x, hist_y = 1001, 600
    style = "vector"

    fig, ax = plt.subplots(figsize=(8,6))
    fig.tight_layout()
    # writer = rp.viz.save.FFmpegMP4(fig, fps, "video.mp4")
    # writer.open()
    for samps in sink(num_samps):
        ts = time.perf_counter()
        snips = rp.utils.stft.stft(samps, nfft, overlap, window)
        psds = rp.utils.stft.psd(snips, Fs)

        im = rp.viz.rt.Persistent(ax, psds, hist_x, hist_y, style=style)
        ax.set_title(f"{sink.cur_samp/Fs:.2f}s / {sink.percent*100:.2f}%", loc="right")
        ax.set_title(f"Playback {samp_time}:{samp_show}ms", loc="left")
        ax.set_title(f"{snips.shape[1]}*{nfft} FFTs")
        ts = (time.perf_counter()-ts)*1000
        # writer.grab_frame()
        print(f"Loop {sink.percent*100:.2f}%: {ts:.2f}ms")
        plt.pause((samp_show-ts)/1000)
        # plt.show()
        # ax.cla()
    # writer.close()

if __name__ == "__main__":
    main()
