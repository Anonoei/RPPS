import rpps as rp

sink = rp.serial.File("f32","data/fm_rds_250k_1Msamples.iq")
meta = rp.Meta(Fs=250e3, cf=99.5e6)
sweep_time, display = 10, 50 # ms
num_samps = int((sweep_time/1000)*meta.Fs)
nfft, overlap, vbw = 1024, 0.8, 1000
window = "blackman"

fig, ax = rp.viz.subplots()
fig.tight_layout()
im = ax
for samps in sink(num_samps):
    snips = rp.utils.stft.stft(samps, nfft, overlap, window)
    psds = rp.utils.stft.psd(snips, meta.Fs, vbw)

    im = rp.viz.rt.Persistent(im, psds)
    rp.viz.pause((display)/1000)
