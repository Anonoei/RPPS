import rpps as rp
import numpy as np
import matplotlib.pyplot as plt

from scipy import signal

# From https://pysdr.org/content/rds.html

def main():
    # num_samps = -1
    num_samps = 1024*16
    sink = rp.serial.File("cf64", "data/fm_rds_250k_1Msamples.iq")
    meta = rp.Meta(Fs=250e3, cf=99.5e6)

    mod = rp.mod.FSK()
    shift = rp.filters.ShiftFreq(fo=-57e3) # Frequency shift to RDS
    low_pass = rp.filters.low_pass(101, 7.5e3, meta.Fs) # Lowpass, isolate RDS
    # low_pass = rp.filters.low_pass(101, 3.0e3, meta.Fs) # Lowpass, isolate RDS
    decimate = rp.sample.down.Decimate(10) # Decimate by 10
    resample = rp.sample.re.Polyphase(19, 25) # Resample 25k to 19k

    rrc = rp.filters.ShapingRRC(101, 4)

    sym_sync = rp.sync.Sync(rp.sync.time.MM(0.01, 0.01, 32)) # Symbol sync
    freq_sync = rp.sync.Sync(rp.sync.Costas(8.0, 0.02, order=2)) # Fine Frequency correction

    tcor = 0.0
    fcor = 0.0
    pcor = 0.0

    fig, axs = plt.subplots(1,2, layout="constrained")
    ax_t = axs[0]
    ax_p = axs[1]
    # ax_p = ax

    for samps in sink(num_samps):
        # meta.obj = mod.decode(samps)
        # # FM Demod
        meta.obj = 0.5 * np.angle(samps[0:-1] * np.conj(samps[1:]))

        # # Freq shift
        # meta += shift
        N = len(meta.obj)
        f_o = -57e3 # amount we need to shift by
        t = np.arange(N)/meta.Fs # time vector
        meta.obj = meta.obj * np.exp(2j*np.pi*f_o*t) # down shift

        # # Isolate RDS
        # meta += low_pass
        taps = signal.firwin(numtaps=101, cutoff=7.5e3, fs=meta.Fs)
        meta.obj = np.convolve(meta.obj, taps, 'valid')

        # # decimate by 10
        # meta += decimate
        meta.Fs = 25e3
        meta.obj = meta.obj[::10]

        # # resample to 19k
        # meta += resample
        meta.Fs = 19e3
        meta.obj = signal.resample_poly(meta.obj, 19, 25) # up, down

        # meta.obj = obj

        meta.sps = 16
        # meta += rrc
        # ax_t = rp.viz.eye.IQ(ax_t, meta.obj, meta.sps)
        # ax_t = rp.viz.time.IQ(ax_t, meta.obj)
        # ax_t = rp.viz.freq.psd(ax_t, meta.obj, meta.Fs, meta.cf)

        meta = meta + sym_sync
        tcor = sym_sync.impl.est
        ax_t = rp.viz.phasor(ax_t, meta.obj)

        meta = meta + freq_sync
        fcor = freq_sync.impl.freq
        pcor = freq_sync.impl.pha

        state = f"{sink.percent*100:06.2f}% | tcor: {tcor:09.3e} / fcor: {fcor:09.3e} / pcor: {pcor:09.3e}"
        print(state)
        fig.suptitle(state)
        # ax_t = rp.viz.time.IQ(ax_t, meta.obj)
        N = 50
        for idx in range(0, len(meta.obj)-N, N):
            ax_p = rp.viz.phasor(ax_p, meta.obj[idx:idx+N])
            plt.pause(0.1)
        # plt.show()
        # break
        # ax_p = rp.viz.phasor(ax_p, meta.obj)
        plt.pause(0.2)
        meta.reset()

if __name__ == "__main__":
    main()
