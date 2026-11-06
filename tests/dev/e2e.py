import rpps as rp
import matplotlib.pyplot as plt

# From https://pysdr.org/content/rds.html

def main():
    # num_samps = -1
    num_samps = 1024*16 # Any block size works, the stages keep the stream continuous
    sink = rp.serial.File("cf64", "data/fm_rds_250k_1Msamples.iq")

    # RDS is 1187.5 symbols/s. Decimate and Resample scale sps along with Fs, so sps is 16 at the sync
    meta = rp.Meta(Fs=250e3, sps=250e3/1187.5, cf=99.5e6)

    demod = rp.mod.FMDemod() # FM Demod
    shift = rp.filters.ShiftFreq(fo=-57e3) # Frequency shift to RDS
    low_pass = rp.filters.low_pass(101, 7.5e3, meta.Fs) # Lowpass, isolate RDS
    decimate = rp.sample.down.Decimate(10) # Decimate by 10
    resample = rp.sample.re.Polyphase(19, 25) # Resample 25k to 19k
    matched = rp.filters.ShapingBiphase() # RDS symbols are biphase

    sym_sync = rp.sync.Sync(rp.sync.time.MM(0.05, 0.0, 32, biphase=True)) # Symbol sync
    freq_sync = rp.sync.Sync(rp.sync.Costas(0.132, 0.00932, order=2)) # Fine Frequency correction

    chain = (demod, shift, low_pass, decimate, resample, matched, sym_sync, freq_sync)

    fig, axs = plt.subplots(1,2, layout="constrained")
    ax_t = axs[0]
    ax_p = axs[1]

    for samps in sink(num_samps):
        meta.obj = samps
        for stage in chain:
            meta = meta + stage
        # ax_t = rp.viz.eye.IQ(ax_t, meta.obj, meta.sps)
        # ax_t = rp.viz.freq.psd(ax_t, meta.obj, meta.Fs, meta.cf)

        tcor = sym_sync.state(meta).est
        fcor = freq_sync.state(meta).freq
        pcor = freq_sync.state(meta).pha

        state = f"{sink.percent*100:06.2f}% | tcor: {tcor:09.3e} / fcor: {fcor:09.3e} / pcor: {pcor:09.3e}"
        print(state)
        fig.suptitle(state)
        ax_t = rp.viz.time.IQ(ax_t, meta.obj)
        N = 50
        for idx in range(0, len(meta.obj)-N, N):
            ax_p = rp.viz.phasor(ax_p, meta.obj[idx:idx+N])
            plt.pause(0.1)
        plt.pause(0.2)
        meta.reset()

if __name__ == "__main__":
    main()
