# tdoa: transmitter location from time differences

Locate a transmitter from simultaneous captures at three or more known positions. Framework only: tested on simulated captures, not yet on real radios.

    python projects/tdoa/tdoa.py synth --snr 0 --seconds 20      # self-test: 4 receivers, start offsets, ppm clock error and drift, noise
    python projects/tdoa/tdoa.py measure a.cf64 b.cf64 --fs 8 --cf 407 --freq 407.6125 --seconds 30
    python projects/tdoa/tdoa.py locate sites.json

`locate` reads `{"ref": 0, "receivers": [{"lat": .., "lon": ..}, ..], "tdoa_ns": [0, ..], "sigma_ns": [.., ..]}`, delays relative to receiver `ref` after calibration.

## Procedure
1. Place at least 3 radios around the target (2D fix; 4 for 3D). Share one reference clock between them (GPS-disciplined 10 MHz, or one radio's CLKOUT to the others' CLKIN) and start them together.
2. Capture a span that holds both the target channel and a transmitter at a known position (a TV pilot, a licensed control channel) at the same time, 30-120 s.
3. `measure` the target and the reference for each receiver against receiver 0. `calibrate` turns the reference's known delays into per-receiver start offsets; subtract them from the target delays.
4. `locate` solves the position and gives the 1-sigma error ellipse. Geometry matters (`gdop`): receivers should surround the target.

## What the simulation shows (4 receivers, 5 km scale, 20 s, 12.5 kHz-class signal)
| SNR | position error | stated 1-sigma axes |
|---|---|---|
| +10 dB | 15 m | 98 / 54 m |
| 0 dB | 117 m | 309 / 173 m |
| -10 dB | 1353 m | 901 / 633 m |

Limits not simulated: multipath (dominant in cities), shared-clock phase noise, receivers that cannot be started together. A narrowband channel limits precision; a wider signal (ATSC, LTE) is far better.

Open items: weight the correlation by the signal spectrum, use a wider channel, handle independent clocks (repeat calibration), try real captures.
