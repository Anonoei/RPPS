import rpps as rp
import numpy as np

import matplotlib.pyplot as plt
import time

def main():
    sink = rp.serial.File("f32","data/fm_rds_250k_1Msamples.iq")
    meta = rp.Meta(Fs=250e3, cf=99.5e6)
    sweep_time, display = 10, 50 # ms
    num_samps = int((sweep_time/1000)*meta.Fs)
    nfft, overlap, vbw = 1024, 0.8, 1000
    window = "blackman"

    fig, ax = plt.subplots()
    fig.tight_layout()
    im = ax
    # fps = (1000/display)/4
    # writer = rp.viz.save.FFmpegMP4(fig, fps, "media/example_rt.mp4")
    # writer.open()
    loop_sum = 0
    loop_count = 0
    loop_min = np.inf
    loop_max = -np.inf

    for samps in sink(num_samps):
        ts = time.perf_counter()
        snips = rp.utils.stft.stft(samps, nfft, overlap, window)
        psds = rp.utils.stft.psd(snips, meta.Fs, vbw)

        im = rp.viz.rt.Persistent(im, psds)
        ax.set_title(f"{sink.cur_samp/meta.Fs:.2f}s / {sink.percent*100:.2f}%", loc="right")
        ax.set_title(f"Playback {sweep_time}:{display}ms", loc="left")
        ax.set_title(f"{snips.shape[1]}*{nfft} FFTs")
        ts = (time.perf_counter()-ts)*1000
        # writer.grab_frame()
        print(f"Loop {sink.percent*100:.2f}%: {ts:.2f}ms")
        loop_sum += ts
        loop_count += 1
        if loop_min > ts:
            loop_min = ts
        if loop_max < ts:
            loop_max = ts
        plt.pause((display-ts)/1000)
    # writer.close()
    print(f"{loop_sum/1000:.1f}s total ({loop_count} loops) / {(loop_sum/loop_count):.2f}ms mean per loop")
    print(f"Min: {loop_min:.2f}, Max: {loop_max:.2f}")


if __name__ == "__main__":
    main()
