import rpps as rp
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as manim
import time

class Prop:
    samp_time = 10 # ms
    samp_show = 100 # ms
    Fs = 250_000
    num_samps = int((samp_time/1000)*Fs)

    nfft = 1024
    overlap = 0.8
    window = rp.filters.window.blackman
    hist_x, hist_y = 1001, 600

def rt_sdot(psds):
    return rp.viz.rt.matrix.sdot(Prop.hist_x, Prop.hist_y, psds)

def rt_dot(psds):
    return rp.viz.rt.matrix.dot(Prop.hist_x, Prop.hist_y, psds)

def rt_vec(psds):
    return rp.viz.rt.matrix.vec(Prop.hist_x, Prop.hist_y, psds)

def rt_svec(psds):
    return rp.viz.rt.matrix.svec(Prop.hist_x, Prop.hist_y, psds)

def speed(sink):
    samps = sink.read(Prop.num_samps)
    snips = rp.utils.stft.stft(samps, Prop.nfft, Prop.overlap, Prop.window)
    psds = rp.utils.stft.psd(snips, Prop.Fs)

    count = 100

    results = []
    def test(psds, count, func, name):
        run_time = time.perf_counter()
        for _ in range(count):
            mat, (amp_min, amp_max) = func(psds)
        run_time = (time.perf_counter() - run_time)*1000
        print(f"{name} complete!")
        return (mat, f"{name}: {run_time:.2f}ms / {run_time/count:.2f}ms/per")

    # results.append(test(psds, count, rt_sdot, "sdot"))
    results.append(test(psds, count, rt_svec, "svec"))
    # results.append(test(psds, count, rt_dot, "dot"))
    results.append(test(psds, count, rt_vec, "vec"))

    fig, axs = plt.subplots(len(results),1)
    fig.tight_layout()
    fig.text(0.1, 0.98, f"Itterations: {count}")
    for ax, res in zip(axs, results):
        mat = res[0] / np.max(res[0])
        ax.imshow(mat, cmap=rp.viz.rt.colors.hot,
            origin="lower", vmin=0, vmax=1,
            aspect="auto",
            interpolation="none", resample=False,
            animated=True
        )
        ax.set_title(res[1])
    plt.show()

def main():
    sink = rp.serial.File("f32","/home/ano/sync/dev/anonoei/RPPS/data/fm_rds_250k_1Msamples.iq")
    # speed(sink)
    loop(sink)

def loop(sink):
    fig, ax = plt.subplots(figsize=(8,6))
    fig.tight_layout()
    # fps = (1000/Prop.samp_show)
    # writer = rp.viz.save.FFmpegMP4(fig, fps, "video_vector.mp4")
    # writer.open()
    loop_sum = 0
    loop_count = 0
    loop_min = np.inf
    loop_max = -np.inf

    for samps in sink(Prop.num_samps):
        ts = time.perf_counter()
        snips = rp.utils.stft.stft(samps, Prop.nfft, Prop.overlap, Prop.window)
        psds = rp.utils.stft.psd(snips, Prop.Fs)

        rp.viz.rt.Persistent(ax, psds, Prop.hist_x, Prop.hist_y)
        ax.set_title(f"{sink.cur_samp/Prop.Fs:.2f}s / {sink.percent*100:.2f}%", loc="right")
        ax.set_title(f"Playback {Prop.samp_time}:{Prop.samp_show}ms", loc="left")
        ax.set_title(f"{snips.shape[1]}*{Prop.nfft} FFTs")
        ts = (time.perf_counter()-ts)*1000
        # writer.grab_frame()
        print(f"Loop {sink.percent*100:.2f}%: {ts:.2f}ms")
        loop_sum += ts
        loop_count += 1
        if loop_min > ts:
            loop_min = ts
        if loop_max < ts:
            loop_max = ts
        # plt.pause((samp_show-ts)/1000)
        plt.pause((Prop.samp_show-ts)/1000)
        ax.cla()
    # writer.close()
    print(f"{loop_sum/1000:.1f}s total ({loop_count} loops) / {(loop_sum/loop_count):.2f}ms mean per loop")
    print(f"Min: {loop_min:.2f}, Max: {loop_max:.2f}")


if __name__ == "__main__":
    main()
