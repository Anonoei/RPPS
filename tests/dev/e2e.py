import rpps as rp
import numpy as np
import matplotlib.pyplot as plt

# From https://pysdr.org/content/rds.html

def main():
    num_samps = 1024*8
    sink = rp.serial.File("cf64", "data/fm_rds_250k_1Msamples.iq")
    meta = rp.Meta(Fs=250e3, cf=99.5e6)

    mod = rp.mod.FSK()
    shift = rp.filters.ShiftFreq(fo=-57e3) # Frequency shift to RDS
    low_pass = rp.filters.low_pass(101, 7.5e3, meta.Fs) # Lowpass, isolate RDS
    decimate = rp.sample.down.Decimate(10) # Decimate by 10
    resample = rp.sample.re.Polyphase(19, 25) # Resample 25k to 19k

    sym_sync = rp.sync.Sync(rp.sync.time.MM(0.01, 0.01)) # Symbol sync
    # freq_sync = rp.sync.Sync(rp.sync.Costas(8.0, 0.02, order=2)) # Fine Frequency correction

    fig, ax = plt.subplots(2)
    fig.tight_layout()
    ax_t = ax[1]
    ax_p = ax[0]

    for samps in sink(num_samps):
        fig.suptitle(f"{sink.percent*100:.2f}%, {sink.cur_samp}/{sink.max_samp}")
        meta.obj = mod.decode(samps)
        meta = meta + shift + low_pass + decimate + resample
        meta.sps = 16
        meta = meta + sym_sync
        print(f"sym {sym_sync.impl.est:.3e}")
        # meta = meta + freq_sync
        # print(f"freq: {freq_sync.impl.freq*meta.Fs/(2*np.pi):.3e}, pha: {freq_sync.impl.pha:.3e}")
        print(meta)
        ax_t = rp.viz.time.IQ(ax_t, meta.obj)
        ax_p = rp.viz.phasor(ax_p, meta.obj)
        plt.pause(0.5)
        # ax[0].cla()
        # ax[1].cla()
        meta.reset()

if __name__ == "__main__":
    main()
