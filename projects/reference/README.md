# reference: what a receiver's measurements are worth

Run from the repo root. Each tool measures one of the receive chain's own properties (signals of known frequency or
rate, a second feed, a known level) and prints the pysdrlib call that keeps it. RPPS doesn't measure these itself: it
reads them from the receive context (pysdrlib's `RxContext`), and the signal profile says what a carrier's frequency,
rate and level are worth from them.

| Tool | Measures | Kept as (pysdrlib) | Read by RPPS as |
|---|---|---|---|
| [frequency.py](frequency.py) | the radio's reference error (ppm), an LNB's LO | `set_freq_correction(ppm, accuracy=)`, a chain's `lo` / `lo_accuracy` | the profile's `frequency` step |
| [polarization.py](polarization.py) | which feed each carrier is on, cross-polar leakage, frequency reuse | a chain's `polarization` (per `channel`) | a recording's feed |
| [power.py](power.py) | dBFS to dBm at the antenna port | `set_level_calibration(offset, accuracy=)` | the profile's `level` step |

`reference_io.py` holds what they share: a capture from a file (a `.sigmf` brings its receive context) or the radio
(`--freq`, `--fs`, `--seconds`, `--gain`; `RPPS_RADIO`, `PYSDRLIB_REMOTE`).

## frequency.py

    python projects/reference/frequency.py --freq 751 --fs 15.36                      # the radio: RPPS_RADIO, PYSDRLIB_REMOTE
    python projects/reference/frequency.py --file cap.cf32 --fmt cf32 --fs 15.36 --freq 751

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

## polarization.py

    python projects/reference/polarization.py h.sigmf v.sigmf                    # two recordings, one per feed
    python projects/reference/polarization.py h.sigmf                            # one: the feed its chain names
    python projects/reference/polarization.py --record cap --freq 11700 --fs 20 --names H V   # two channels of one radio

One feed shows one polarization, so a single capture only says which feed it came through. Two feeds received at the same
instants (a dual-output LNB on a B210's two channels) say, per carrier: its C/N on each feed against that feed's own floor
(unequal gains cancel), and how coherent the two copies are. The coherence is measured bin by bin over the inner 80% of
the band, so a delay between feeds doesn't matter and a copy far under the noise is still found. The verdicts:

- **H / V** (one feed): XPD is how far under it the other feed's copy is. "its leakage" marks a copy the other feed's
  survey listed as a carrier of its own. `>` marks a lower bound, when no copy is seen.
- **both**: the same signal on both feeds, within 10 dB: a slanted linear polarization, or a circular one on linear feeds.
- **reuse**: on both feeds but not the same signal: two carriers at one frequency, one per polarization.

`--record` sets each channel's chain (`Chain(port, channel=k, polarization=...)`, keeping the radio's own LO) and writes
`PREFIX.<name>.sigmf` with each channel's receive context, so the recordings name their feeds. Synthetic checks (40 seeds): a −25 dB
leak reads 24.9–26.4 dB, a −35 dB one (under the noise) 34.5–36.5 dB.

## power.py

    python projects/reference/power.py floor --nf 8 --freq 751 --fs 10        # no equipment: the input terminated
    python projects/reference/power.py tone --dbm -60 --freq 751 --fs 10      # a signal generator's tone at the centre
    python projects/reference/power.py yfactor --enr 15 --on on.sigmf --off off.sigmf --fs 10

Prints dBm at the port = dBFS + offset, and `sdr.set_level_calibration(offset, accuracy=)`. pysdrlib keeps it only for
the gains and frequency it was set at, and the profile's `level` step then gives each carrier in dBm.

- **floor**: with nothing on the input, the floor is kT0 (−174 dBm/Hz) raised by the noise figure. A datasheet NF is
  a few dB off, and that is the accuracy. The gain must be high enough for the analog noise, not the ADC's, to set the
  floor: 6 dB more gain should lift the floor by about 6 dB.
- **tone**: a known level at the port (a generator through a known cable). The accuracy is the generator's.
- **yfactor**: a noise source of known ENR, on and off. Y gives the noise figure, then the floor gives the offset, to a
  few tenths of a dB.

EIRP isn't measured here: it needs the antenna's gain and the path's loss as well.
