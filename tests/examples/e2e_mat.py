import rpps as rp
import rpps.viz
import numpy as np
import matplotlib.pyplot as plt

# From https://pysdr.org/content/rds.html
# The same RDS receiver as e2e.py, with no loops: rp.sync.Acquire finds the symbol rate, timing, modulation and frequency
# from the stream, rp.sync.time.Mat takes the symbols with a matrix matched filter (no MM), and rp.sync.phase.FFCarrier
# corrects frequency and phase feed-forward (no Costas).
# RDS is biphase: Acquire finds the 2375 Hz chip rate (twice the 1187.5 bit rate), so the output is BPSK chips, two per bit.

def main():
    num_samps = 1024*16 # Any block size works, the stages keep the stream continuous
    sink = rp.source.File("cf64", "data/fm_rds_250k_1Msamples.iq", Fs=250e3, sps=250e3/1187.5, cf=99.5e6)

    demod = rp.wave.FMDemod() # FM Demod
    shift = rp.signal.filters.ShiftFreq(fo=-57e3) # Frequency shift to RDS
    low_pass = rp.signal.filters.low_pass(101, 7.5e3, 250e3) # Lowpass, isolate RDS
    decimate = rp.signal.sample.down.Decimate(10) # Decimate by 10
    resample = rp.signal.sample.re.Polyphase(19, 25) # Resample 25k to 19k

    acquire = rp.sync.Acquire() # Detect baud/modulation/cfo from the first blocks
    sym_sync = rp.sync.time.Mat() # Matched filter + timing, at the detected sps
    carrier = rp.sync.phase.FFCarrier() # Frequency and phase from the detection

    pre = demod + shift + low_pass + decimate + resample + acquire # Everything up to the symbol sync
    post = sym_sync + carrier

    fig, axs = plt.subplots(1,3, layout="constrained", figsize=(12,6))
    ax_t = axs[0]
    ax_e = axs[1]
    ax_p = axs[2]

    track = None # Es/N0 over the last symbols, once there is a detection
    pts = None
    shown = np.zeros(0, dtype=np.complex64) # The symbols on the phasor

    for mid in sink(num_samps) + pre:
        out = mid + post
        det = out.det

        # Eye of what the symbol sync sees, centred on where it samples
        ax_e.clear()
        ax_e.set_title("Eye (sync input)")
        if det is None:
            state = f"{sink.percent*100:06.2f}% | acquiring"
        else:
            if track is None:
                pts = rp.analysis.modulation.candidates([det.mod])[det.mod]
                track = rp.analysis.quality.Track(pts)
            if len(mid) >= 4*round(mid.sps):
                phase = rp.viz.eye.sync_phase(out.state(sym_sync), len(mid))
                if phase is not None:
                    rp.viz.eye.IQ(ax_e, mid.data, mid.sps, phase, beta=det.beta, lock=det.M) # Through the matched filter
            q = track.update(out.data, sps=det.sps)
            state = f"{sink.percent*100:06.2f}% | {det.mod} {det.baud:0.1f} Bd | " + (f"{q}" if q else "measuring")
        print(state)
        fig.suptitle(state)

        if len(mid):
            ax_t = rp.viz.freq.psd(ax_t, mid.data, mid.Fs, mid.cf)
        if len(out):
            # shown = np.concatenate((shown, out.data))[-1000:]
            ax_p = rp.viz.phasor(ax_p, out.data, points=pts)
        plt.pause(0.01)

if __name__ == "__main__":
    main()
