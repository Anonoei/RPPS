import rpps as rp
import numpy as np
import matplotlib.pyplot as plt

def main():
    num_samps = 256
    sink = rp.serial.File("cf64", "data/fm_rds_250k_1Msamples.iq")
    meta = rp.Meta(Fs=250e3, cf=99.5e6)

    samps = sink.read(num_samps)
    low_pass = rp.filters.low_pass(101, 7.5e3, meta.Fs) # Lowpass, isolate RDS
    low_pass = low_pass.filt

    taps = 40
    t = np.arange(-taps//2, taps//2)

    lp_exp = rp.filters.designer.lowpass(taps, "exp")
    lp_sqr = rp.filters.designer.lowpass(taps, "sqr")
    lp_sinc = rp.filters.designer.lowpass(taps, "sinc")

    print(f"exp: {np.sum(lp_exp)}")
    print(f"sqr: {np.sum(lp_sqr)}")
    print(f"sinc: {np.sum(lp_sinc)}")

    hp_exp = rp.filters.designer.highpass(taps, "exp")
    hp_sqr = rp.filters.designer.highpass(taps, "sqr")
    hp_sinc = rp.filters.designer.highpass(taps, "sinc")

    print(f"exp: {np.sum(hp_exp)}")
    print(f"sqr: {np.sum(hp_sqr)}")
    print(f"sinc: {np.sum(hp_sinc)}")


    fig, ax = plt.subplots(3, 2)
    fig.tight_layout()

    ax[0,0].plot(t, lp_exp, "o")
    ax[1,0].plot(t, lp_sqr, "o")
    ax[2,0].plot(t, lp_sinc, "o")

    ax[0,1].plot(t, hp_exp, "o")
    ax[1,1].plot(t, hp_sqr, "o")
    ax[2,1].plot(t, hp_sinc, "o")

    for i in range(3):
        for j in range(2):
            ax[i,j].set_xlim(t[0], t[-1])
            ax[i,j].grid(True)
    plt.show()
    exit()


    fig, ax = plt.subplots(2,2)
    fig.tight_layout()

    ft = ax[1,0]
    ff = ax[1,1]
    sf = ax[0,0]
    sc = ax[0,1]

    ft.set_title("Filter Time")
    ff.set_title("Filter Freq")
    sf.set_title("Samples Freq")
    sc.set_title("Filtered Freq")

    ft.plot(low_pass)
    ff.plot(np.fft.fftshift(np.abs(np.fft.fft(low_pass))))

    sf.plot(np.fft.fftshift(np.abs(np.fft.fft(samps))))
    samps = np.convolve(samps, low_pass, mode="same")
    sc.plot(np.fft.fftshift(np.abs(np.fft.fft(samps))))
    plt.show()


if __name__ == "__main__":
    main()
