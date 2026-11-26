# reference: what a receiver's frequencies are worth

Run from the repo root. Measures the radio's reference error (ppm) and an LNB's LO error from signals of known
frequency or rate in a capture, and prints the pysdrlib calls that keep them. RPPS doesn't measure these itself: it
reads them from the receive context (pysdrlib's `RxContext`: `ppm_accuracy`, a chain's `lo_accuracy`), and the signal
profile's `frequency` step says what a carrier's frequency and symbol rate are worth from them.

## reference.py

    python projects/reference/reference.py --freq 751 --fs 15.36                      # the radio: RPPS_RADIO, PYSDRLIB_REMOTE
    python projects/reference/reference.py --file cap.cf32 --fmt cf32 --fs 15.36 --freq 751

Finds the LTE cells in the capture (the bands the survey's boxes make up, 1.4 to 20 MHz wide, searched for a PSS and SSS
at their middle). Each gives two references: its carrier on LTE's 100 kHz raster, and the sample clock against its 5 ms
half-frames (base stations are GPS-locked to 0.05 ppm). A least-squares fit, with priors for what nothing measured (a
radio's own reference within 25 ppm, an LNB's LO within a MHz), gives both errors and their standard errors. A raster
is used only once the rest pins the error under a quarter of its step; at 3.5 GHz one cell alone can't, and the raster
points it could be are listed instead.

Then keep the result where the radio is described:

    sdr.set_freq_correction(-13.941, accuracy=0.066)       # the radio's reference (what it already had, plus the residual)
    lo = 9750012345, lo_accuracy = 150                     # an LNB's chain in a pysdrlib frontends file (illustrative values)

The module (`FrequencyReference`, `RateReference`, `rate`, `fit`, `lte`, `find`, `calibrate`) takes other references
too: a beacon at a known frequency, a symbol rate a standard fixes (ATSC's 10.762238 MBd).

On air (2026-11-26): the HackRF at 751 MHz, one LTE cell (PCI 25): -13.941 +- 0.066 ppm in 2 s; the half-frames alone
-14.6 +- 0.5 ppm.
