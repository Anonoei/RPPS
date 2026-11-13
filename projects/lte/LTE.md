# LTE: cell search and what comes next

Working notes. The code is `projects/lte/lte_cell.py` (cell search), `projects/lte/lte_pbch.py` (PBCH -> MIB), `projects/lte/lte_sib.py` (PCFICH/PDCCH/PDSCH -> SIB1), `projects/lte/lte_turbo.py`, `projects/lte/lte_rrc.py`; capture with `record.py`, gain with `gain_search.py`.

Specs used (3GPP, fetched from `3gpp.org/ftp/Specs/archive/36_series/<n>/<n>-j30.zip`, Rel-19 "j30"; the .docx text extracts with a ~10-line zipfile+regex script, but **equations are lost** in extraction, so formulas are from memory and confirmed on the data): TS 36.211 (6.6 PBCH, 6.10.1 CRS, 6.3.4.3 transmit diversity, 7.2 Gold sequence, 6.11 PSS/SSS), TS 36.212 (5.1.1 CRC, 5.1.3.1 tail-biting code, 5.1.4.2 rate matching, 5.3.1 BCH, table 5.3.1.1-1 CRC masks), TS 36.331 (6.2.2 MasterInformationBlock, PHICH-Config, the ASN.1), TS 36.213 (7.1 PDSCH, table 7.1.7.2.1-1).

## Result so far

Band 13 (10 MHz, nominal centre 751.0 MHz), captured 0.5 s at 15.36 MS/s, gain amp on / LNA 16 / VGA 8 (28 dB SNR, no clipping,
no artifacts from the gain), tuned to 753 MHz so the radio's DC spike falls away from the sync signals:

| quantity | value |
|---|---|
| PSS root | 29, so N_ID2 = 1 (ratio 297 against 25 and 34 for the other two roots; chance is ~3-5) |
| SSS | N_ID1 = 8 in **101 of 101** occurrences, metric 1.00 vs 0.49 for the runner-up (chance 0.05); half-frame alternates 1, 0, 1, 0 ... |
| **cell ID (PCI)** | 3 x 8 + 1 = **25** |
| carrier | 751.01196 MHz = +15.9 ppm from 751.0 |
| sample clock | -16.1 ppm (from the spacing of 101 PSS occurrences: 76798.76 samples, scatter 0.34) |
| cyclic prefix | coefficient 0.73 (noise 0.05); residual offset +1962 Hz, spread 72 Hz |

The carrier offset and the sample-clock error are the same oscillator (the radio's), seen in two ways.

## PBCH / MIB (band 13, same capture): decoded

`python projects/lte/lte_pbch.py data/lte_b13.cf64 --fs 15.36 --cf 753 --carrier 751.0` (22 s; `--selftest` encodes/decodes synthetic MIBs).

| quantity | value |
|---|---|
| frames used | 50 (each subframe 0 in the 0.5 s capture; every one decodes alone) |
| transmit antenna ports | **4** (CRC mask 0101..., on 50 of 50; equalizer agrees: QPSK-ness 0.97 as 4-port SFBC-FSTD vs 0.13 as 1 port and -0.34 as 2) |
| dl-Bandwidth | **n50 = 50 RB = 10 MHz** (agrees with the survey and the sample rate) |
| PHICH | duration normal, resource 1 |
| system frame number | 551 ... 600 across the 50 frames, +1 per frame without a miss (8 MSBs from the MIB, 2 LSBs from which quarter of the scrambling sequence decodes) |
| schedulingInfoSIB1-BR | 13 (non-zero: **the cell serves LTE-M / eMTC devices**; 0 would mean no SIB1-BR) |
| systemInfoUnchanged-BR, partEARFCN, spare | 0, 0, 0 |
| symbol spread across the QPSK diagonals | 0.073 (0 = perfect) |

### How it works

1. **Where**: subframe 0 begins 7 symbols (one slot) minus the PSS before the PSS: `S0 = pss_pos - pss_off` (the same constant as in the cell search,
   6656 at 15.36 MS/s), taken from the PSS occurrences whose SSS said "half-frame 0". PBCH = slot 1, symbols 0-3, subcarriers `6*N_RB-36 .. +71`.
2. **Window**: FFT each symbol 8 samples early (inside the cyclic prefix; ISI-free) and remove the resulting phase ramp `exp(+j 2 pi f 8/N)`.
3. **Channel estimate**: the CRS value is `(1/sqrt2)[(1-2c(2m)) + j(1-2c(2m+1))]`, Gold `c_init = 2^10 (7(ns+1)+l+1)(2*PCI+1) + 2*PCI + 1`, index `m' = m + 110 - N_RB`,
   on subcarriers `6m + (v + PCI mod 6) mod 6` (v: port 0: 0 at l=0, 3 at l=4; port 1 the opposite; port 2: 3 (ns mod 2); port 3: 3 + 3 (ns mod 2)).
   Port 0/1 CRS are in symbols 0 and 4 of both slots, port 2/3 only in symbol 1. LS at the CRS, linear in frequency, straight line in time.
4. **Resource elements**: the PBCH skips the REs of *all four* ports' CRS whatever is actually transmitted: in symbols 0 and 1 every third subcarrier
   (`(k - PCI) mod 3 == 0`), none in symbols 2 and 3: 48 + 48 + 72 + 72 = 240 QPSK symbols = 480 bits per frame. Order: increasing k, then l.
5. **Equalization** (6.3.4.3): 1 port: `conj(h0) r`. 2 ports: Alamouti on RE pairs. 4 ports: SFBC on ports (0,2) over the first RE pair of a quartet and ports (1,3)
   over the second: `r_a = h0 x0 - h2 x1*`, `r_b = h0 x1 + h2 x0*`, `r_c = h1 x2 - h3 x3*`, `r_d = h1 x3 + h3 x2*`.
6. **Soft bits and scrambling**: bit 0 = positive; XOR with Gold `c_init = PCI`, one 480-bit quarter per frame (which quarter = SFN mod 4; try all four).
7. **Rate matching** (5.1.4.2): D = 40 bits per coded stream, 32-column sub-block interleaver (24 dummy bits first), circular buffer of the three streams one after
   another, 120 real bits repeated to 1920. A single frame holds 4 laps of the buffer, so each frame decodes alone and a quarter can be soft-combined.
8. **Tail-biting rate-1/3 K=7** (generators 133, 171, 165 octal; MSB = current input; register preloaded with the last six input bits): Viterbi over three laps
   of the circle, keep the middle lap.
9. **CRC-16** (g = D^16 + D^12 + D^5 + 1, zero start) of the 24 MIB bits, XORed with the 16 received parity bits, must equal the mask for 1 / 2 / 4 ports (all 0 / all 1 / 0101...).

### Statistic per assumption, and a trap

- **QPSK-ness** `-Re(sum z^4) / sum |z|^4` of the equalized symbols: 1 for clean QPSK, ~0 or negative when the channel estimate or the precoding is wrong. It needs no
  decoding and told the 4-port answer immediately.
- **The CRC alone does not tell the demodulation hypothesis.** Demodulated as 1 or 2 ports, a 4-port signal *still passed the CRC on all 50 frames*: the
  wrongly-combined symbols still lean the right way, and a rate-1/3 code with 16-fold repetition rescues them at this SNR. So the CRC proves "these are the MIB
  bits" and, through its mask, the port count; the QPSK-ness (or the symbol spread) is what proves the equalizer. Use both.
- **Self-test only proves the encoder and decoder agree with each other.** Whether the generators, interleaver and bit order match the standard is only settled by the CRC
  passing on a real signal (chance 1 in 65536 per mask), and it did, first try.
- The remaining uncertainty I could not check from the extracted text (equations): exact 4-port precoding matrix (confirmed by the data), rate-matching start offset k0 = 0
  (confirmed by the data).

## SIB1 (band 13, same capture): decoded

`python projects/lte/lte_sib.py data/lte_b13.cf64 --fs 15.36 --cf 753 --carrier 751.0` (about 30 s). Chain: PCFICH -> PDCCH blind decode for SI-RNTI -> DCI 1A -> PDSCH (turbo)
-> CRC-24A -> ASN.1 (UPER) with `lte_rrc.py`. Modules: `lte_sib.py` (control channel + PDSCH), `lte_turbo.py` (turbo codec), `lte_rrc.py` (ASN.1).

| quantity | value |
|---|---|
| SIB1 subframes in the capture | 25 (subframe 5 of every even frame, SFN 552 ... 600) |
| DCI with SI-RNTI 0xFFFF found | **25 / 25** (aggregation 4 at CCE 0, 27 payload bits = format 1A, localized, MCS 7) |
| CFI (PDCCH symbols) | 1 in 23 subframes, 2 in one, 3 in one (each time the DCI found only under that CFI) |
| redundancy version in the DCI | 0, 2, 3, 1, 0, 2, 3, 1 ... exactly the SIB1 pattern (36.321 5.3.1: rv = ceil(1.5 k) mod 4, k = (SFN/2) mod 4): not an input, so an independent check |
| PDSCH | 2-6 PRBs per transmission, at a different place each time (21+6, 21+3, 45+5, 48+2, 36+6, 12+6 ...); TBS 224 bits (I_TBS 7, N_PRB^1A = 2) |
| PDSCH CRC-24A | **25 / 25 pass**, every redundancy version, on one transmission each (no soft combining needed) |
| payload | the same 28 bytes in all 25: `484c46901070101d11029181822103108ac21a2692cba60000000000` |

Decoded SIB1 (ASN.1 from TS 36.331):

| field | value |
|---|---|
| PLMN | MCC 311, MNC 480 (a US operator code; the public MCC/MNC list assigns 311-480 to Verizon Wireless) |
| tracking area code | 0x0701 = 1793 |
| cell identity | 0x01D1102 = 1904898 (28 bits = 20-bit eNodeB ID + 8-bit cell ID): eNodeB 0x01D11 = 7441, cell 2 |
| cell barred / intra-frequency reselection / CSG | notBarred / allowed / false |
| q-RxLevMin | -64 (x2 = -128 dBm), offset 1 |
| **freqBandIndicator** | **13** (agrees with the 751 MHz carrier I found independently) |
| scheduled SI messages | #1 every 16 frames: SIB3 (SIB2 is always in the first message); #2 every 64 frames: SIB5 + SIB8; #3 every 64 frames: SIB4; SI window 20 ms; value tag 2 |
| extensions | IMS emergency support, eDRX allowed, hyper-SFN present |

### The pieces, and what to know about each

- **Subframe grid** (`subframe_grid`): all 14 symbols FFT'd 8 samples early with the ramp removed; CRS channel estimates for all four ports on every RE (LS at the CRS, linear in
  frequency, a straight line in time). Subframe 5's start = the PSS occurrence of half-frame 1 minus the same PSS offset as before.
- **PCFICH** (36.211 6.7): 4 REGs of symbol 0 at `k = 6 (PCI mod 2 N_RB) + floor(i N_RB/2) 6`, 16 QPSK symbols, scrambling `c_init = (floor(ns/2)+1)(2 PCI+1) 2^9 + PCI`,
  CFI codewords (36.212 table 5.3.4-1) period 3. CFI = argmax of the correlation with the three codewords; margin 2.8-3.6x.
- **Control-region REGs** (6.2.4): symbol 0 and (4 ports) symbol 1 use 6-subcarrier REGs with the 4 non-CRS REs (every third subcarrier is CRS); symbols 2, 3 use 4 consecutive subcarriers.
  **PHICH** (6.9.3) takes 3 x ceil(Ng N_RB/8) REGs of symbol 0 at `n_i = (PCI + m' + floor(i n0/3)) mod n0` over the n0 REGs that are not PCFICH (Ng from the MIB).
  **PDCCH** gets the rest, filled *time-first* (for each frequency position, symbols 0..CFI-1).
- **PDCCH decode** (6.8.2, 6.8.5): equalize each REG as one SFBC-FSTD quartet, quadruplets cyclically shifted by PCI in transmission (so undo `w(j) = wbar((j - PCI) mod M)`),
  permuted by the 32-column sub-block interleaver applied to quadruplets, 9 REGs per CCE, scrambling `c_init = floor(ns/2) 2^9 + PCI`.
  Blind search in the common search space: aggregation 4 at CCE 0, 4, 8, 12 and 8 at 0, 8; every payload size K = 10..60 through the tail-biting decoder; accept when
  `CRC16(payload) XOR parity == 0xFFFF` (SI-RNTI) or 0xFFFE (P-RNTI): about 300 trials at 2^-16 each is a 0.5% false-alarm chance in total, and it found the right one in 25 of 25.
- **DCI 1A** (36.212 5.3.3.1.3), 50 RB, FDD: flag 1, VRB flag 1, RIV 11 bits, MCS 5, HARQ 3, NDI 1, RV 2, TPC 2 = 26 bits (+1 pad bit = 27: it is on the list of sizes that get one
  bit so as not to collide with another format's size). With an SI-RNTI the TPC's low bit chooses the TBS column N_PRB^1A = 2 or 3 and the NDI doubles as the gap bit.
  RIV -> (start, length) as in 36.213 7.1.6.3.
- **PDSCH** (36.211 6.3, 36.213 7.1.7): localized PRBs; REs in increasing k over the PRBs, then l, from symbol CFI to 13; CRS of all four ports skipped (symbols 0, 1, 4, 7, 8, 11),
  and the central 72 subcarriers of the PSS/SSS symbols (5 and 6 of slot 0, subframes 0 and 5). Scrambling `c_init = RNTI 2^14 + floor(ns/2) 2^9 + PCI`; QPSK; four-port transmit diversity
  decoded exactly as the PBCH/PDCCH quartets. TBS from the 36.213 table 7.1.7.2.1-1.
- **Turbo code** (36.212 5.1.3.2, 5.1.4.1): K = TBS + 24; QPP interleaver (the 188 (K, f1, f2) triples parsed from the spec text, each verified to be a permutation); streams of K + 4;
  sub-block interleaver with the *different* permutation for the second parity stream (pi(k) = (P(k div R) + 32 (k mod R) + 1) mod K_pi); circular buffer v0 | v1 v2 interleaved;
  k0 = R (2 ceil(N_cb/8R) rv + 2) with N_cb = K_w (the soft buffer is not limiting for system information). Max-log-MAP decoder, 8 iterations, scale 0.75, tails ignored.
- **CRC-24A** polynomial 0x864CFB; the transport block plus its CRC has remainder 0.

### Traps

- **Spec text extraction loses the equations and the tabs.** Formulas that matter (PHICH mapping, PCFICH positions, 4-port precoding matrix, the PDCCH mapping steps) were read by
  converting the .docx to PDF with LibreOffice (`soffice --headless --convert-to pdf`) and rendering the page with `pdftoppm -png`. Tables of numbers survive text extraction
  (QPP table, TBS table, CFI codewords, CRC masks). One typo in the spec document's TBS table: I_TBS 6, N_PRB 1 reads 328 (should be 88).
- **My memory of the TBS columns was off by a shift** (I had I_TBS 7 / N_PRB 2 = 176; the table says 224). The CRC would have caught it, but only because I read the table.
- **The allocation changes from one SIB1 transmission to the next** (21+6, then 45+5, then 12+6 ...): do not cache the PRBs, read each DCI. Allocations that cross the central 6 PRBs
  must skip the PSS/SSS REs.
- **asn1tools cannot compile 3GPP's parameterized `SetupRelease {X}`**: expand it textually (see `lte_rrc.py`). `pip install asn1tools` into a venv: it is not an RPPS dependency.
- **QPSK-ness is low (0.13, 0.31) on one allocation (PRB 36-41, SFN 576/578) yet the CRC still passes**: the rate is low (6 PRBs for 224 bits), so partial equalization still decodes.
  Probably a channel-estimate issue away from the centre; worth a look when decoding larger messages.

## System information messages (SIB2, 3, 4, 5, 8): decoded

Same run with `--asn eutra.asn` (build it once with `lte_rrc.py build`). SIB1's `schedulingInfoList` gives each SI message's period and the 20 ms window; message n (1-based) starts at
x = (n-1) w ms, so SFN mod T = floor(x/10) and first subframe x mod 10 (36.331 5.2.3). The capture (SFN 551-600) holds message 1 three times (SFN 560, 576, 592) and messages 2 and 3
once (SFN 578 and 580), each as a PDCCH with SI-RNTI in subframe 0 of the window. All five decode (CRC-24A) from one transmission each, 77 s for the whole run.

| message | content | what it says |
|---|---|---|
| 1 (TBS 328, MCS 10) | **SIB2** | PRACH config index 19, root sequence 313, 44 preambles; PUCCH nRB-CQI 5; 64QAM uplink enabled; paging cycle 128 frames; **reference-signal power 21 dBm per RE**, p-b 1; UE timers (T300 400 ms, T301 1000 ms, T310 2000 ms, T311 3000 ms); access-barring skip for MMTel voice/video; voice-service cause indication |
| 1 | **SIB3** | intra-frequency cell reselection: q-Hyst 2 dB, serving-frequency priority 0, q-RxLevMin -64, t-Reselection 1 s |
| 2 (TBS 296, MCS 9) | **SIB5** | two other LTE carriers (below) |
| 2 | SIB8 | CDMA2000 1xRTT parameters (CS fallback support, search window 8, synchronous system time) |
| 3 (TBS 56, MCS 1) | **SIB4** | intra-frequency excluded cells PCI 91 and 396; CSG PCI range 472..479 |

**SIB5's inter-frequency carriers: EARFCN 2560 and 2050.** EARFCN -> frequency with 36.101 table 5.7.3-1: band 5 (N_DL 2400 = 869.0 MHz): 869.0 + 0.1 x (2560 - 2400) = **885.0 MHz**;
band 4 (N_DL 1950 = 2110.0 MHz): 2110.0 + 0.1 x (2050 - 1950) = **2120.0 MHz**. These are the centres of two of the plateaus found independently in the spectrum survey (880.6-889.5 MHz,
10 MHz wide; 2111.0-2129.1 MHz, 20 MHz wide), so the survey and the decode agree on where the operator's other carriers are.

Things learnt on the way:
- **SI messages use QPSK whatever the MCS**: MCS 9 and 10 (which would be 16QAM for ordinary data) decode as QPSK with this TBS table (36.213 7.1.7.1: Qm = 2 for SI-RNTI, P-RNTI, RA-RNTI).
- **They use the whole carrier**: 24-27 PRBs, against 2-6 for SIB1; SIB1's allocation moves around, the SI allocations too.
- **A DCI can be found twice**: the aggregation-8 candidate at CCE 0 also contains the aggregation-4 copy (the rate-matched bits repeat), so deduplicate by DCI bits.
- 0.5 s is enough only because the windows happen to fall inside it. For a full survey of a cell's SIBs (SIB6-SIB16 can be on 128-512 frame periods) capture 5 s or more.

## Other carriers, the scanner, and traffic

**Band 5 (885 MHz) and band 4 (2120 MHz, 20 MHz wide)**, found from SIB5's EARFCNs and decoded with the same chain:

| carrier | PCI | bandwidth | ports | band / PLMN / TAC | eNodeB / cell | SIB1 |
|---|---|---|---|---|---|---|
| 751.0 MHz (b13) | 25 | 50 RB | 4 | 13 / 311-480 / 0x0701 | 7441 / 2 | 49/49 |
| 885.0 MHz (b5) | 25 | 50 RB | 4 | 5 / 311-480 / 0x0701 | 7441 / 29 | 49/49 |
| 2120.0 MHz (b4) | 25 | 100 RB | 4 | 4 / 311-480 / 0x0701 | 307441 / 22 | 42/50 |

A 28-bit cell identity is a 20-bit eNodeB ID plus an 8-bit cell ID (an earlier note here had the split wrong). Bands 13 and 5 are the same site (eNodeB 7441); band 4 is eNodeB 307441.
Each cell's SIB5 lists the other two (EARFCN 5230 = 751.0 MHz, 2560 = 885.0, 2050 = 2120.0). The 20 MHz carrier needs 30.72 MS/s, a 20 MS/s radio cannot: capture at 20 MS/s and
`rp.signal.sample.resample_poly` by 192/125; the outer RBs are attenuated by the radio's baseband filter but the CRS-based equalizer simply weights them down, and everything decoded.

**`lte_scan.py`** does the whole job: `--scan` sweeps the bands (20 MS/s tiles, gain stepped down per tile until nothing clips), finds plateaus of LTE widths (4.5 / 9 / 13.5 / 18 MHz occupied),
then for each: the full gain search, capture, resample to the LTE rate, cell search, MIB, SIB1, and with `--traffic` the load measurement. `--carrier MHz --bw N` skips the sweep;
`--file` re-analyses a saved capture. Details that mattered:
- The capture is centred 2 MHz (1.5 for 20 MHz) away from the carrier so the DC spike misses PSS/SSS, on the side that keeps the strongest neighbour outside the baseband filter; the "empty band" for
  the gain search is the quietest 0.6 MHz of the sweep inside the capture (a first version guessed the capture's edge and picked gain 0 on a weak carrier).
- The radio streams only during a capture (it warns about hundreds of millions of dropped samples otherwise).
- **A validity gate**: PSS correlation always returns a "best" PCI, even on noise. A carrier counts only if the SSS agrees across at least half of the PSS occurrences (real cells: 90-100%, noise: 2%) and the MIB CRC passes.
  Without it the scan "found" five cells with 100% load that were noise.

Scan of 728-756, 869-894, 1930-1995, 2110-2155 MHz (8 candidates, 1 s each): three real cells (the table above); the other five (739.1, 1932.6, 1986.2, 1992.6, 2146.0 MHz) were PSS correlations on noise or too weak
to decode (SSS agreement 2%): plateaus that are real signals (739 is a flat 9 MHz, +13 dB) but not strong enough at this location to read with this antenna. Worth trying with a better antenna or a longer capture.

**Traffic** (what "load" means here, and what is deliberately not done). For every subframe: the mean power of the CRS REs, and for each PRB the mean power of its data REs (symbols 3-13, CRS REs left out). A PRB counts as *active* when its data power exceeds
0.12 x the CRS power (-9 dB; PDSCH sits around -3..-6 dB relative to the CRS, idle PRBs at the noise level, -25..-35 dB). The 6 central PRBs (PSS/SSS/PBCH/SIB1) and the PRBs around the radio's own DC spike
are left out (the DC spike showed up as a permanent fake load of 5% until masked). CFI per subframe from the PCFICH; PDCCH grants addressed to the public RNTIs only: SI-RNTI and P-RNTI (paging) with the K = 27 / 13 DCI sizes
in the subframes where paging can occur (0, 4, 5, 9). **Not done on purpose**: paging records (they carry subscriber identifiers), PDSCH to individual users (private, encrypted at higher layers), blind decoding of other RNTIs.

| carrier | load (PRB utilisation) | CFI | paging messages in 1 s | notes |
|---|---|---|---|---|
| 751 MHz (b13) | 3% | 73% 1 symbol | 12 | bursts of a few PRBs; a regular six-PRB stripe in the centre (broadcast) |
| 885 MHz (b5) | 2% in the first capture, 35% in the second (bursts of ~15 ms of 5-15% load rising to ~50% for ~200 ms) | 98% 1 symbol | 17 | the same cell at two moments minutes apart |
| 2120 MHz (b4) | 15% mean, 4% median, 47% at the 90th percentile | 96% 1 symbol | 0 | bursty |

Heat maps (PRB x time, PRB data power / CRS power) are written by `--keep DIR`. Estimated false-alarm rate of the paging detector: 12 trials x 2/65536 per occasion = 3.7e-4, so well under one false paging hit in a capture.

## Where the carriers are (spectrum survey, 700-900 and 1920-2200 MHz)

Flat plateaus of LTE bandwidths (9 MHz occupied = a 10 MHz carrier), level over the floor:

| occupied | probable carrier | level |
|---|---|---|
| 734.6-743.5 MHz | 10 MHz, band 12/17 | +10 dB |
| 746.3-755.5 MHz | **10 MHz, band 13 (done)** | +27 dB |
| 880.6-889.5 MHz | 10 MHz, band 5 | +11 dB |
| 1960.4-1973.3 MHz | 15 MHz, band 2 | +19 dB |
| 2111.0-2129.1 MHz | 20 MHz, band 4/66 | +21 dB |
| 1946-1959, 2175-2180 MHz | two ~5 MHz, one 5 MHz | +8 .. +16 dB |

The 700-900 MHz band clipped heavily at LNA 32 / VGA 24 (10% of samples): strong signals. Use a lower gain there.

## The method (same as ATSC: one statistic per assumption)

1. **PSS** (TS 36.211 6.11.1): three Zadoff-Chu roots 25, 29, 34, 62 subcarriers either side of DC, last OFDM symbol of subframes 0 and 5.
   Correlate at 1.92 MS/s (the central 1.4 MHz) over a grid of frequency offsets, fold on 5 ms. The offset has to be searched: a 16 ppm clock at
   751 MHz is 12 kHz, nearly a whole 15 kHz subcarrier. 5 kHz steps are enough (the peak is ~15 kHz wide).
2. **PSS timing of each occurrence** at the full rate. The sample clock error moves occurrences by ~1 sample per 5 ms, so locate each one near
   where the previous predicted it (+-30 samples), then fit a line: that gives the sample-clock error directly.
3. **Cyclic prefix** for the fractional-subcarrier offset: the first Ncp samples of a symbol repeat its last Ncp; the phase of the correlation
   is 2 pi df N / fs. Needs 0.1 s to be steady; spread 72 Hz.
4. **SSS** (6.11.2): two interleaved m-sequences give N_ID1 and the half-frame. Equalise the SSS symbol (the one before the PSS) with the PSS's
   channel estimate and correlate with all 168 x 2 hypotheses. This is **coherent across two OFDM symbols 71 us apart**, so the frequency
   offset must be right to ~100 Hz: a wrong offset by a few kHz turns the whole thing to chance. That is why step 3 matters.

## What went wrong on the way (and how it showed)

- **A constant that was a symbol short.** `pss_off` (the PSS symbol's useful start from the slot start) must be (80 + 1024) + 5 x (72 + 1024) + 72 = 6656 at
  15.36 MS/s; I had 5632 (forgot symbol 0's useful part). Symptoms: the cyclic-prefix coefficient was 0.29 not 0.73, the offset it gave (-5.2 kHz)
  was impossible given the search grid, and the SSS vote was scattered (49 / 22 / 21 of 101). The measurement that found it: scan the assumed slot
  start and see the coefficient peak a constant 72 samples (one prefix length) away from where the PSS said.
- **The cyclic prefix pins timing only modulo one symbol**, since the long first prefix (80) differs from the others (72) by 8 samples, and a window
  one prefix length off also scores well on a short span. Use the PSS for exact timing; use the CP only for the frequency offset.
- **The grid drifts**: the slot start moved 4945 -> 4853 over 0.39 s (the clock error again), so any single fixed grid is wrong at the far end of a capture.

## Next, in order

1. ~~PBCH / MIB~~, ~~PCFICH / PDCCH / SIB1~~, ~~SI messages~~ done (above).
2. ~~The other carriers~~ done (above); the original plan: targets from SIB5: **885.0 MHz (band 5, 10 MHz)** and **2120.0 MHz (band 4, 20 MHz)**, same chain, with per-band gain tuning (always, per the artifact rule).
   The 20 MHz carrier needs 30.72 MS/s but a radio limited to 20 MS/s cannot: capture at 20 MS/s (covers +-10 MHz, the carrier is +-9) and resample to 30.72 MS/s
   (`rp.signal.sample.re.Polyphase`, ratio 192/125); everything downstream (FFT 2048, 100 RB) is already parameterised by `nfft`. The same SIB1 reading then says which operator and band each is.
3. **Combine retransmissions** (soft-combine the four RVs of SIB1; HARQ) for weak cells: `lte_turbo.dematch(..., d)` already accumulates.
4. **Cell survey on neighbour PCIs** (SIB4 excluded cells 91, 396; a cell search at 751 MHz finds only the strongest cell: look for other PSS roots at lower levels by cancelling the first).
5. **Into RPPS**: an OFDM demodulator with CRS channel estimation, PSS/SSS searcher, tail-biting and turbo decoders, a control-channel blind decoder, and the 36.331 ASN.1 build step.
   Functions written to be lifted: `gold`, `crs_values`, `conv_encode`/`viterbi_tailbiting`, `crc16`, `crc24a`, `turbo_encode`/`turbo_decode`/`rm_index`, `control_region`, `diversity4`, `pdsch_res`.

## What it will and will not decode

PSS, SSS, PBCH and SIBs are broadcast control information that every phone reads to attach; they are public by design. User data (PDSCH to a
particular phone) is scrambled with a per-user RNTI and encrypted at higher layers: not something this work will or should recover.
