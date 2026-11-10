# fm_radio: FM broadcast receiver

Run from the repo root. Difficulty: beginner. Needs a supported SDR and `pw-play` or `paplay` for sound. Radio access goes through [../common/radio_io.py](../common/radio_io.py).

## fm_rx.py: live FM receiver

    python projects/fm_radio/fm_rx.py --scan                 # what is on the air (88-108 MHz)
    python projects/fm_radio/fm_rx.py --scan --rds           # ... and read each station's RDS (name, type, text, clock)
    python projects/fm_radio/fm_rx.py 105.1                  # lock to a station and play it
    python projects/fm_radio/fm_rx.py 105.1 --rds            # ... and decode RDS (name, text, clock)
    python projects/fm_radio/fm_rx.py 105.1 --no-audio --rds # decode only
    python projects/fm_radio/fm_rx.py 105.1 --wav out.wav    # record the audio too

What it does: surveys around the dial frequency (`rp.dsp.survey`) to find the carrier, tunes 300 kHz below it
(keeping the radio's DC spike out of the signal), streams at 2.4 MS/s, and channelizes to the 480 kS/s MPX signal.
From the MPX it takes mono (L+R), stereo (L-R, switched in when the 19 kHz pilot is locked), and RDS (57 kHz).

- **Lock**: the carrier is tracked from the demodulator's DC offset (AFC); the stereo carrier from the pilot (a PLL in `PilotPLL`).
- **Status line** every 2 s: carrier, AFC error, load (processing time / real time, keep it under ~90%), pilot level,
  RDS state and Es/N0. `DROPPED` means processing can't keep up; lower the load or use `--mono`/no `--rds`.
- **Gain**: `--gain` (a fraction 0-1 of the radio's range). The default (0.42; pysdrlib splits it across the radio's gain stages) came from a gain sweep on the FM band; strong
  local stations may want less, weak ones more, but watch for ADC clipping.
- **Scan with RDS**: tunes to each station found and listens up to `--rds-time` seconds (default 12; 4 s if there is no
  RDS). Reports PI, name, programme type, RadioText and clock when sent. Many stations rotate or scroll the 8-character
  name (song titles, slogans), flagged `PS rotates`; the RadioText is usually the better identifier. Weak RDS is
  reported only when enough groups agree on one PI, so noise doesn't show up as a station.
- **De-emphasis**: `--tau 75` (Americas) or `--tau 50` (elsewhere).

`rds_link.py` is the RDS link layer (block sync by syndrome, groups, PI / PS / RadioText / clock). RPPS recovers the
symbols; this turns them into text.
