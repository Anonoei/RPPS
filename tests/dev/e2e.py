import rpps as rp
import matplotlib.pyplot as plt

# From https://pysdr.org/content/rds.html

def main():
    # num_samps = -1
    num_samps = 1024*16 # Any block size works, the stages keep the stream continuous
    # RDS is 1187.5 symbols/s. Decimate and Resample scale sps along with Fs, so sps is 16 at the sync
    sink = rp.serial.File("cf64", "data/fm_rds_250k_1Msamples.iq", Fs=250e3, sps=250e3/1187.5, cf=99.5e6)

    demod = rp.mod.FMDemod() # FM Demod
    shift = rp.filters.ShiftFreq(fo=-57e3) # Frequency shift to RDS
    low_pass = rp.filters.low_pass(101, 7.5e3, 250e3) # Lowpass, isolate RDS
    decimate = rp.sample.down.Decimate(10) # Decimate by 10
    resample = rp.sample.re.Polyphase(19, 25) # Resample 25k to 19k
    matched = rp.filters.ShapingBiphase() # RDS symbols are biphase

    sym_sync = rp.sync.time.MM(0.05, 0.0, 32, biphase=True) # Symbol sync
    freq_sync = rp.sync.Costas(0.132, 0.00932, order=2) # Fine Frequency correction

    rx = demod + shift + low_pass + decimate + resample + matched + sym_sync + freq_sync

    fig, axs = plt.subplots(1,2, layout="constrained")
    ax_t = axs[0]
    ax_p = axs[1]

    for out in sink(num_samps) + rx:
        # ax_t = rp.viz.eye.IQ(ax_t, out.data, out.sps)
        # ax_t = rp.viz.freq.psd(ax_t, out.data, out.Fs, out.cf)

        tcor = out.state(sym_sync).est
        fcor = out.state(freq_sync).freq
        pcor = out.state(freq_sync).pha

        state = f"{sink.percent*100:06.2f}% | tcor: {tcor:09.3e} / fcor: {fcor:09.3e} / pcor: {pcor:09.3e}"
        print(state)
        fig.suptitle(state)
        ax_t = rp.viz.time.IQ(ax_t, out.data)
        N = 50
        for idx in range(0, len(out)-N, N):
            ax_p = rp.viz.phasor(ax_p, out.data[idx:idx+N])
            plt.pause(0.1)
        plt.pause(0.2)

if __name__ == "__main__":
    main()
