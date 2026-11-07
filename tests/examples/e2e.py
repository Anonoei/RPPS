import rpps as rp
import rpps.viz
import numpy as np
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

    pre = demod + shift + low_pass + decimate + resample + matched # Everything up to the symbol sync
    post = sym_sync + freq_sync

    bpsk = np.array([1.0, -1.0]) # What the Costas loop locks to
    track = rp.analysis.quality.Track(bpsk) # Es/N0 over the last symbols
    shown = np.zeros(0, dtype=np.complex64) # The symbols on the phasor

    fig, axs = plt.subplots(1,3, layout="constrained", figsize=(12,6))
    ax_t = axs[0]
    ax_e = axs[1]
    ax_p = axs[2]

    with rp.viz.save.FFmpegMP4(fig, 2, "video.mp4") as writer:
        for pre_out in sink(num_samps) + pre:
            # Eye of what the symbol sync sees: the matched-filter output, at sps
            ax_e.clear()
            ax_e.set_title("Eye (sync input)")
            ax_t = rp.viz.freq.psd(ax_t, pre_out.data, pre_out.Fs, pre_out.cf)
            out = pre_out + post
            if len(pre_out) >= 4*round(pre_out.sps):
                # Centre the eye on where the sync samples
                phase = rp.viz.eye.sync_phase(out.state(sym_sync), len(pre_out))
                rp.viz.eye.IQ(ax_e, pre_out.data, pre_out.sps, phase, width=0.5, derotate=2) # BPSK, one lobe wide

            tcor = out.state(sym_sync).est
            fcor = out.state(freq_sync).freq
            pcor = out.state(freq_sync).pha

            q = track.update(out.data, sps=pre_out.sps)
            state = f"{sink.percent*100:06.2f}% | tcor: {tcor:09.3e} / fcor: {fcor:09.3e} / pcor: {pcor:09.3e} | " + (f"{q}" if q else "measuring")
            print(state)
            fig.suptitle(state)
            if len(out):
                # shown = np.concatenate((shown, out.data))[-1000:]
                ax_p = rp.viz.phasor(ax_p, out.data, points=bpsk)
            plt.pause(0.01)
            writer.grab_frame()

if __name__ == "__main__":
    main()
