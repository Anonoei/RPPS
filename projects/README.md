# projects

Worked examples of using RPPS (with [pysdrlib](https://github.com/Anonoei/pysdrlib) for the radio) on real signals. Each folder is one project with its own README. They are plain scripts: run them from the repo root (`python projects/ble/ble_scan.py ...`), every script answers `--help`.

## Where to start

| difficulty | project | what it shows |
|---|---|---|
| beginner | [fm_radio/](fm_radio/) | FM broadcast receiver with stereo and RDS: the shortest path from samples to sound |
| beginner | [ble/](ble/) | Bluetooth LE advertising: GFSK demodulation, whitening, CRC-24, PDU parsing |
| beginner | [adsb/](adsb/) | 1090 MHz ADS-B / Mode S: pulse-position decoding, CRC, aircraft position |
| intermediate | [wifi/](wifi/) | 802.11b DSSS and 802.11a/g OFDM receivers, MAC parser, access-point scanner |
| intermediate | [spectrum/](spectrum/) | wideband sweeps, channel occupancy, allocation lookup, live sweep monitor, gain tuning |
| intermediate | [tdoa/](tdoa/) | locating a transmitter from time differences between synchronised receivers (simulation-tested) |
| advanced | [atsc/](atsc/) | ATSC 8VSB digital TV: equaliser, trellis, Reed-Solomon, MPEG transport stream, live monitor |
| advanced | [lte/](lte/) | LTE downlink: cell search, MIB, SIB1 and SI messages, carrier scanner |
| advanced | [p25_dmr/](p25_dmr/) | P25 phase 1 / 2 and DMR control and voice decoding, IMBE vocoder, live monitor |

Order within a level does not matter. `common/` holds the helpers every project shares (below), `tests/` the tests.

## Setup

Any radio pysdrlib supports works. Install RPPS and pysdrlib (and `pw-play` or `paplay` for sound).

- **Which radio:** `RPPS_RADIO=<pysdrlib driver>`, or `--radio`; unset uses the first radio pysdrlib finds.
- **Remote radio:** `PYSDRLIB_REMOTE=tcp://host:5555` (a `pysdrlib server` on the machine with the radio); every script that opens the radio through `common/radio_io.py` uses it. The monitors also take `--remote`.
- **Gain** is one number everywhere (`--gain`, the ladders): a fraction 0-1 of the radio's own range (pysdrlib `set_rx_gain`), so the same fraction is the same place in the range on any radio.
- **Two receive channels:** `RPPS_CHANNELS=2` makes `wifi_scan`, `ble_scan`, `vhf_scan`, `voice_scan`, `lte_scan` and `gain_search.py` test two gains per capture on a radio with two receive channels; it needs the same signal on both inputs. `RPPS_ANTENNA` picks channel 0's input.

## common/

- `radio_io.py`: open the radio, set gains, grab samples, count clipping, `capture_clean` (the highest non-clipping gain for bursty signals).
- `gain_search.py`: choose the gain for a band.
- `record.py`: capture to a file.

### gain_search.py

    python projects/common/gain_search.py 507 --sig 500.5 505.5 --vac 508 513    # tune MHz, the signal's band, an empty band

Tries gains from 0 to 1 in steps of 0.05 and scores each: in-band SNR, minus a large penalty for ADC clipping, minus a penalty for lines in the empty band that most settings don't show (overload and intermodulation the gain caused). A line present at nearly every setting is the band's own, or the radio's spur, and isn't counted. Prints the answer as a `--gain G` flag. Do this for each band: a gain that suits FM is not right for UHF.

### record.py

    python projects/common/record.py 507 20 0.6 data/capture.cf64 --gain 0.8   # MHz, MS/s, seconds, file

complex64 (`rp.source.File("cf64", ...)`). Reports peak level and the fraction of clipped samples.

## Conventions

- **Gain:** tune it for the band to remove artifacts (clipping, the intermodulation lines overload makes), not just to maximise SNR. `gain_search.py` scores both.
- **Clock error:** a radio's reference is typically tens of ppm off and drifts, so every tool here estimates the carrier offset instead of assuming it.
- **Captures** are complex64 (`.cf64`) at the stated sample rate, centred so the DC spike misses the signal. Keep them in an untracked `data/`.
- **Standards:** each project's README says which parts of which standard were used; when a decode stalls, read the standard.
- **Imports:** a script starts with a one-line bootstrap that puts every project folder on `sys.path` via `_paths.py`, so any script can import any other module and runs from any directory. Module names are unique across folders.
- **Tests:** `python -m pytest projects/tests` (every script compiles, imports and answers `--help`; synthetic-signal tests for ADS-B, ATSC and the sweep).
- Listen-only: nothing here transmits, and encrypted traffic is not decoded.

See [TODO.md](TODO.md) for what is missing.
