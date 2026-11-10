# wifi: passive 2.4 / 5 GHz Wi-Fi decoding

Everything here is listen-only. A radio at 20 MS/s sees one 20 MHz Wi-Fi channel at a time; these programs find the bursts, decode the PHY and the
MAC frame, check the CRC-32, and report access points from their beacons. Only access points are listed. Client devices' addresses are not collected
or printed (probe requests are only counted), and payloads are never decoded (data frames are mostly encrypted anyway).

Run from the repo root. Needs a supported SDR, numpy and scipy. `wifi_scan.py` uses the shared radio helpers in `projects/common/radio_io.py`.

| file | what it is |
|---|---|
| `wifi_scan.py` | the scanner: captures channels, runs both receivers, prints the access-point table |
| `wifi_dsss.py` | 802.11b receiver: 1 and 2 Mbit/s DSSS (Barker), long preamble. This is how most 2.4 GHz access points send beacons |
| `wifi_ofdm.py` | 802.11a/g receiver: 6 to 54 Mbit/s OFDM. Required on 5 GHz; also decodes what 802.11n/ac/ax devices send in legacy format (beacons, ACKs) |
| `wifi_mac.py` | MAC frame parser: frame types, beacon information elements (SSID, channel, rates, RSN/WPA, HT/VHT/HE capability) |

## Usage

    python projects/wifi/wifi_scan.py                                  # 2.4 GHz channels 1-13, 2 s each
    python projects/wifi/wifi_scan.py --channels 1 6 11 --seconds 5 --keep captures/
    python projects/wifi/wifi_scan.py --band5 --channels 36 40 44 48 149 153 157 161
    python projects/wifi/wifi_scan.py --files "captures/wifi_ch{}.cf64"   # analyse saved captures (no radio)
    python projects/wifi/wifi_dsss.py capture.cf64 --fs 20             # one receiver on one file
    python projects/wifi/wifi_ofdm.py capture.cf64 --fs 20

Captures are complex64 at 20 MS/s centred on the channel (2407 + 5 x ch MHz; 5000 + 5 x ch MHz on 5 GHz).

### Gain
`gain_search.py` scores SNR against an empty band, which means nothing for bursty traffic (a first attempt picked gain 0 on three busy channels and saw nothing). The scanner
instead takes the **highest gain on a ladder that does not clip over a 0.3 s window** (clipping threshold 2e-5), per channel. It varies a lot: amp on / LNA 24 / VGA 8 on most
channels here, LNA 32 on the busiest, and amp off / LNA 0 close to a strong transmitter. Strong nearby transmitters clip first, so a quiet channel next to one is captured at a lower gain than it could stand.


## How the receivers work, and what each step's check is

**DSSS** (`wifi_dsss.py`, IEEE 802.11b-1999 clause 18): resample to 44 MS/s (4 samples per chip); find the symbol timing as the offset that maximizes despread energy over 100 symbols of the
SYNC field; despread with the 11-chip Barker code `+1 -1 +1 +1 -1 +1 +1 +1 -1 -1 -1`; differential detection (DBPSK; DQPSK at 2 Mbit/s); the carrier offset (a radio's reference is typically ~15 ppm = 36 kHz = 0.23 rad per symbol off at 2.4 GHz)
is estimated as half the angle of the squared differential symbols over the SYNC; timing tracked every 40 symbols by early/prompt/late comparison; descramble (z^-7 + z^-4 + 1, self-synchronising); find the
SFD `1111 0011 1010 0000` (rightmost bit sent first); PLCP header SIGNAL/SERVICE/LENGTH with CRC-16 CCITT; PSDU bytes LSB first; CRC-32 (FCS).
Checks (each at chance level when the assumption is wrong): the SFD after a run of ones (1.5e-5), the header CRC-16 (1.5e-5), the frame FCS (2e-10).
- Yield: of the PLCP headers that pass (92 in channel 11), 25 pass the FCS (6 of 11 short frames, 19 of 81 long ones). The rest decode with a few bit errors: a long frame has perfect symbol margins yet
  one corrupted byte in the middle. Cause (not proven): the capture and the radio's baseband filter cut the 22 MHz DSSS signal, which leaves ISI the rectangular chip template does not remove. Beacons repeat
  every ~100 ms, so an AP still shows up many times per capture; a pulse-shaped matched filter or an equalizer is the improvement to make. A few bursts also contain two frames back to back.
- **Not implemented:** the 5.5 and 11 Mbit/s CCK rates and the short preamble (no beacon uses them; data does).

**OFDM** (`wifi_ofdm.py`, IEEE 802.11a-1999 clause 17): STF delay-16 autocorrelation to find the packet and the coarse carrier offset (alias-free to +-625 kHz); cross-correlation with the known LTF symbol,
two peaks 64 samples apart; fine offset from the two LTF copies; per-subcarrier channel estimate from the LTF; the SIGNAL symbol (BPSK 1/2: 4 rate bits, reserved, 12-bit length, even parity, 6 tail zeros);
per data symbol: FFT 4 samples early inside the guard interval (phase ramp removed), equalize, fit the four pilots' phase against subcarrier index (common phase + timing slope, pilot polarity `p_n` from the scrambler
with all-ones seed), demap (max-log, Gray, 64QAM included), deinterleave (two permutations), depuncture (2/3 and 3/4), soft Viterbi (133/171 octal), descramble (seed recovered from the first seven SERVICE bits, which are zero), CRC-32.
Checks: STF autocorrelation (~1.0 for a real STF, ~0.1 for noise), SIGNAL parity and tail bits (1 in 128), a requirement that the burst be at least as long as the frame it announces (the first version
accepted ~16 false SIGNAL fields from non-OFDM bursts that claimed 1000+ byte frames), and the FCS.
- Verified against the standard's own text: LTF sequence, pilot polarity sequence (all 127 values), pilot values, rate-bit table (see References). Verified on air: control frames and **beacons at 6 Mbit/s** (2.4 and 5 GHz,
  carrier offset -25 kHz at 2.4 GHz and about +40 kHz at 5 GHz), all with valid FCS.
- **Two receiver bugs that hid most frames, and how they showed:** (1) *Timing*: packet start was taken from the STF plateau edge, which wanders by +-40 samples on long frames, while the long-training-field search looked only
  +-40 samples around it. Symptom: strong 20 dB bursts with a perfect STF and no decodable SIGNAL. The LTF correlation itself (two peaks 64 apart, 6-9x the median) was unambiguous: now the LTF pair search over the
  first 860 samples is the primary timing, and the STF is only checked and used for the coarse carrier offset. (2) *SIGNAL parity*: I required bits 17-23 (the parity bit and the six tail bits) all zero, which rejects every SIGNAL
  whose parity bit is 1, about half of all frames. The parity bit makes bits 0-17 sum to even; only bits 18-23 are zero. Symptom (found by printing the decoded SIGNAL of a failing burst): rate 6 Mbit/s, length 343 bytes, burst
  duration matching it exactly, parity "odd". Together these took channel 48 from 1 to 117 valid frames.
- The SIGNAL check is weak on its own: the terminated Viterbi forces the tail bits to zero, leaving parity and rate validity, which random bursts pass about one time in five. The burst-length consistency check and the frame's CRC-32 are the real tests.
- **Not verified:** the punctured rates (3/4 and 2/3) and 64-QAM are implemented but have not passed an FCS yet (see Status: 6, 12 and 24 Mbit/s have). HT/VHT/HE frames (802.11n/ac/ax data) are not decoded: only their legacy preamble is; that is why beacons and ACKs from modern devices work.

**MAC** (`wifi_mac.py`, 802.11-2020 clause 9): frame control (type/subtype), addresses, beacon body (timestamp, interval, capabilities, information elements). Security is read from the RSN element
(group cipher, pairwise ciphers, AKM suites: PSK, SAE, 802.1X, OWE) and the WPA1 vendor element; standards from the HT (45), VHT (191) and HE (extension 35) capability elements.

## References (the standards, free copies)

IEEE 802.11 is paywalled in its current form, but the original supplements are widely mirrored and still describe these PHYs exactly:
- [IEEE Std 802.11a-1999 (OFDM PHY, clause 17)](https://pdos.csail.mit.edu/archive/decouto/papers/802.11a.pdf), also [here](https://gtas.unican.es/files/docencia/CD/practicas/ieee_802.11a.pdf). Used for: LTF sequence (eq. 8), SIGNAL field and rate table (table 80), interleaver (eqs. 15-16), pilot values and polarity sequence (eq. 25), scrambler (figure 113), convolutional code and puncturing.
- [IEEE Std 802.11b-1999 (DSSS/CCK PHY, clause 18)](https://pdos.csail.mit.edu/archive/decouto/papers/802.11b.pdf), also [here](https://simson.net/ref/1999/802.11b-1999.pdf). Used for: long PLCP format, SFD (18.2.3.2), CRC-16 (18.2.3.6), scrambler, Barker sequence, DBPSK/DQPSK.
- [IEEE Std 802.11-2007](https://people.iith.ac.in/tbr/teaching/docs/802.11-2007.pdf) (the merged base standard) lists clause 9 (MAC frames) and the information elements; that mirror returned an error page when I fetched it, so the MAC parser follows the structure from memory of 802.11-2020: **verify element layouts there before relying on rare ones**.
- Background: [VOCAL 802.11a white paper](https://vocal.com/wp-content/uploads/2012/05/80211a_wp1pdf.pdf).

The downloaded PDFs were converted with `pdftotext -layout` and used as text only.

## Traps
- A radio's reference can be ~15 ppm off, which is 36 kHz at 2.4 GHz: both receivers estimate and remove it (the STF alone is good to +-625 kHz, so it is found; the DSSS differential receiver only needs the rotation per symbol).
- 802.11 sends bytes **LSB first**, and the SFD is written MSB-first in the standard with the *rightmost* bit sent first.
- SIGNAL's parity and tail pass by chance 1 time in 128, so a few hundred non-OFDM bursts always produce false positives: check the burst length against the announced frame length.
- The radio's DC spike sits at the capture centre, which for Wi-Fi is the channel centre; subcarrier 0 is unused in OFDM and the OFDM equalizer weights weak subcarriers down, but I have not measured how much SNR it costs.
- The 2 s default is chosen for beacons (every ~100 ms): 20 per AP. For weak APs or rare frames capture longer.

## Status and ideas
- 2.4 and 5 GHz access points are found and described with the Wi-Fi antenna. Not decoded: 802.11n/ac/ax *data* (only their legacy preambles, so beacons and control frames work), CCK 5.5/11 Mbit/s, 40/80 MHz channels beyond the one 20 MHz slice captured.
- OFDM rates with a passing FCS so far: **6 (BPSK 1/2: 76 frames), 12 (QPSK 1/2: 13) and 24 Mbit/s (16-QAM 1/2: 87)**. The punctured rates (9, 18, 36, 54: code rate 3/4; 48: 2/3) and 64-QAM are implemented but have never passed an FCS: untested, so the depuncturing and 64-QAM demapper could still hide a bug.
- Improve DSSS yield (about a quarter of long frames pass the FCS: a pulse-shaped matched filter or an equalizer for the truncated 22 MHz signal).
- 802.11n/ac/ax: parse HT-SIG / VHT-SIG-A / HE-SIG-A to decode the data portion.
- The Bluetooth LE advertising channels (2402, 2426, 2480 MHz) showed strong narrow signals in the wide survey: a BLE advertising decoder (GFSK, 1 Msym/s, access address 0x8E89BED6, whitening by channel) fits here next.
- Into RPPS: the OFDM receiver is the first complete one; `viterbi_r12`, the interleaver and the demapper are written to be lifted out.
