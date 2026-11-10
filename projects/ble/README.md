# ble: Bluetooth LE advertising

Listen-only. A single radio cannot follow a BLE connection (it hops over 37 data channels), but advertising sits on three fixed channels: **37 = 2402, 38 = 2426, 39 = 2480 MHz**.
These programs capture them, find packets, check the CRC-24, and summarize who is advertising. Only advertising packets are read.

Run from the repo root. Needs a supported SDR, numpy and scipy. The radio helpers are in `projects/common/radio_io.py`.

| file | what it is |
|---|---|
| `ble_scan.py` | the scanner: captures the three channels, decodes, prints a summary and a table of advertisers |
| `ble_adv.py` | the receiver: GFSK demodulation, access address, dewhitening, CRC-24, advertising PDU header |
| `ble_parse.py` | PDU parsing: addresses and address kinds, AD structures, manufacturer / service labels (Apple Continuity subtypes, Eddystone, Exposure Notification, Tile, Fast Pair ...) |

## Usage

    python projects/ble/ble_scan.py                       # 6 s per channel
    python projects/ble/ble_scan.py --seconds 30 --keep captures/
    python projects/ble/ble_scan.py --files "captures/ble_{}.cf64"      # saved captures (4 MS/s complex64 centred on the channel)
    python projects/ble/ble_scan.py --hide-names          # do not print local names (shown by default)
    python projects/ble/ble_scan.py --addresses           # print full device addresses
    python projects/ble/ble_adv.py captures/ble_37.cf64 --channel 37

**What it prints**: per channel the packets with a valid CRC-24; packet types; the number of distinct advertiser addresses by kind (public, random static, resolvable private, non-resolvable);
what the advertisers say they are (manufacturer, Apple Continuity message type, known service UUIDs); a table of the strongest advertisers with a 4-hex-digit hash of the address as an id, their address kind,
packet count, strength (dB over the capture's noise floor, uncalibrated), the channels they were heard on, PDU types, labels and, unless `--hide-names`, the local name.
Addresses are hashed unless `--addresses`: the table is meant to describe the neighbourhood, not to identify devices.
**Resolvable private addresses rotate about every 15 minutes**, so the count of distinct addresses overestimates the number of devices (phones and earbuds use them).


## How the receiver works, and each step's check

The standard is the Bluetooth Core Specification v6.0, [Vol 6 Part B, Link Layer Specification](https://www.bluetooth.com/wp-content/uploads/Files/Specification/HTML/Core-60/out/en/low-energy-controller/link-layer-specification.html) (free HTML). The values below were read from it:
- Packet: preamble (8 bits; 0x55 when the access address's LSB is 0), access address **0x8E89BED6**, PDU (2-byte header + payload), CRC. All fields LSB first.
- **Whitening** of PDU and CRC: 7-bit LFSR x^7 + x^4 + 1; position 0 = 1, positions 1-6 = the channel index (37, 38, 39) MSB first.
- **CRC-24**: polynomial x^24 + x^10 + x^9 + x^6 + x^4 + x^3 + x + 1, preset 0x555555 (position 0 = least significant bit) for advertising PDUs, over the PDU bits in transmitted order, sent MSB first (position 23 first).
- Modulation: GFSK 1 Msym/s, index 0.5 (+-250 kHz); a 1 is the higher frequency.

Receiver: capture at 20 MS/s, decimate to 4 MS/s (4 samples per symbol), find bursts by energy (60 us to 3 ms), FM discriminator (`angle(x[n] conj(x[n-1]))`), remove the burst's median frequency (the radio's ~36 kHz
offset at 2.4 GHz is 14% of the deviation), smooth over one symbol, slice at each of the four sample phases and look for the access address (at most 2 wrong bits out of 32, preceded by an alternating
preamble), de-whiten the header, read the length, de-whiten the rest, check the CRC-24. A packet counts only if the CRC passes (chance 6e-8). The first run passed the CRC on the first attempt,
which also confirms the bit orders and the LFSR/CRC conventions above (they are easy to get subtly wrong: there is no free parameter left to tune).

## Gain
Strong Wi-Fi next to channel 38 (2422 and 2427 MHz) clips the 20 MS/s ADC even though it is outside the 4 MS/s channel, so the capture uses `radio_io.capture_clean`: the highest gain that does not clip over a 1 s check window, then if the
capture itself still clipped more than 1e-4 it steps down and captures again. In the first run channel 38 clipped (4.6e-4) at amp on / LNA 24 / VGA 8 and still decoded; the rule gives LNA 16 there.

## Not done / ideas
- LE Coded PHY and 2M PHY (secondary channels), extended advertising's auxiliary packets (the `ADV_EXT_IND` header is recognized but its extended header and AUX packets on the data channels are not followed): these need the
  channel map from the primary packet and a hop-following receiver.
- Connections (data channels): out of reach for a single radio without the connection parameters (and the payloads are usually encrypted).
- A more complete company-identifier table (the SIG's assigned-numbers list: only the common ones are included; unknown ones print as hex).
- Compare antennas: advertisers are weak and short, so the count of packets and advertisers per second is a good figure of merit.
