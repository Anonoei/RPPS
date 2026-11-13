# adsb: 1090 MHz ADS-B / Mode S receiver

Run from the repo root. Usage is also in the docstring of `adsb.py`.

    python projects/adsb/adsb.py --tune                           # try gains, keep the one with the most CRC-valid messages
    python projects/adsb/adsb.py --seconds 30 --gain 0.67
    python projects/adsb/adsb.py data/adsb/a.cf64                 # a saved 2 MS/s capture (cf64 at 1090 MHz)

Finds Mode S frames by their preamble, slices the pulse-position bits, keeps DF17 / DF18 extended squitters whose 24-bit CRC is zero, and groups them by ICAO address:
callsign (type codes 1-4), barometric altitude (9-18), CPR position from an even and an odd frame within 10 s, ground speed and track (19).

**Status:** tested on synthetic captures (`projects/tests/test_scripts_adsb.py`: published example messages, pulse-position modulated with noise). Not decoded yet: DF11 / DF4 / DF5 replies (their CRC carries the interrogator id) and single-bit error correction.
