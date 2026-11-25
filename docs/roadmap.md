# Roadmap

What is in each part of RPPS, and what is not yet.

- [ ] [Source](https://github.com/Anonoei/RPPS/tree/main/src/rpps/source)
    - [X] File, memory, radios (pysdrlib), WAV out
    - [ ] (Linux only) tun/tap
    - [X] Socket (UDP, TCP)
    - [X] SigMF (capture metadata in and out), WAV in
    - [X] `rpps tx`: a profile's transmission to a file
    - [ ] Transmit to a radio
- [ ] [Filters](https://github.com/Anonoei/RPPS/tree/main/src/rpps/signal/filters)
    - [X] Pulse Shaping (rrc, rc, halfsine, gaussian, rect)
    - [X] Window functions
    - [X] Designed in a profile (lowpass, bandpass... in Hz), channelizer, resampler, spectral inversion
- [X] [Sample](https://github.com/Anonoei/RPPS/tree/main/src/rpps/signal/sample)
- [ ] [Sync](https://github.com/Anonoei/RPPS/tree/main/src/rpps/sync)
    - [X] Frequency
    - [X] Phase (M-th power, decision directed on any constellation, pilots), by block estimation tracked across windows
    - [X] Time (Oerder & Meyr; the matched filter and an energy-maximum timing in one, `timing: mat`), AGC by block estimation
    - [X] Per-sample loops, second class (`sync.loops`: Costas, Mueller & Muller, Gardner, timing locked loop, AGC, LMS, CMA)
    - [X] SNR before lock (`sync.quality`: from moments, a known sequence, the cyclic spectrum, the empty band)
    - [X] Frames: preambles among symbols, named words and Zadoff-Chu roots (any width, any rotation), delimiter words among bits (fixed or header-driven length, polarity), repeating markers
    - [X] Equalizers: block least squares (decision directed or blind), OFDM pilots and cyclic prefix; LMS and CMA as loops
    - [X] Per-burst acquisition (`sync.Burst`: each burst a stream of its own; `sync.Bursts`: many carriers, placed by a time plan), preambles (`sync.Preamble`: known waveform or Schmidl-Cox, carrier offset)
    - [X] Tone / pilot PLL (`sync.ToneLock`: FM stereo, ATSC pilot)
    - [X] LTE cell search and resource grid (`grid.lte`: PSS/SSS, sample-clock tracking, channel of each antenna port)
    - [X] NR cell search and SS/PBCH tracking through LEO Doppler and Doppler rate (`grid.nr`)
    - [X] A snapshot lock for programs that seek and retune (`sync.Lock`: find once, follow any later block, say when stale)
    - [ ] Decision-feedback equalizer
- [ ] [Symbol and wave](https://github.com/Anonoei/RPPS/tree/main/src/rpps/symbol)
    - [X] PSK, QAM, APSK, ASK, any point set from a file; differential coding; spreading (Barker, Gold, 802.15.4)
    - [X] FSK, GFSK, MSK, C4FM (frequency pulses), O-QPSK, OFDM, AM, SSB, analog FM
    - [ ] Coherent CPM (GMSK, SOQPSK by MLSE) in the data path
    - [ ] VSB / PAM (ATSC 8VSB), chirp spread spectrum, space-time coding
- [ ] [Code](https://github.com/Anonoei/RPPS/tree/main/src/rpps/code)
    - [ ] Block
        - [X] Repetition
        - [X] Hamming (decoded on its syndrome trellis)
        - [X] Any code bit by bit: parity equations, expressions of the message bits, appended checks, layouts
        - [X] BCH
        - [X] Reed-Solomon
        - [X] LDPC (layered or flooding min-sum; DVB-S2 tables, quasi-cyclic matrices: IEEE 802.11n)
        - [X] Golay, cyclic codes by generator polynomial, product codes (DMR BPTC), turbo (LTE), rate matching (LTE)
        - [X] 5G NR polar (with its rate matching, CRC-aided list decoding) and NR LDPC (both base graphs, rate matching)
        - [X] Polar (successive cancellation, list, CRC-aided)
        - [X] Turbo product codes (Chase-Pyndiah)
        - [X] Reed-Muller
        - [X] Reed-Solomon dual basis (CCSDS)
        - [ ] LDPC: CCSDS AR4JA, DVB-S2X
        - [X] Turbo: CCSDS (rates 1/2 to 1/6), its permutation a definition
        - [ ] Turbo: UMTS, DVB-RCS duo-binary
    - [X] Convolutional
        - [X] Viterbi (hard and soft)
        - [X] BCJR / trellis (log-MAP, max-log-MAP)
        - [X] Trellis codes on levels (ATSC, P25)
        - [X] Trellis-coded modulation on constellation points
    - [X] Soft decisions (demodulator LLRs into the decoders)
- [ ] [Scramblers](https://github.com/Anonoei/RPPS/tree/main/src/rpps/code/scram)
    - [X] fibonacci
    - [X] galois
    - [X] V.35
    - [X] Line codes: NRZI, NRZ-M, Manchester, inversion; sync markers (CCSDS ASM, preambles) attached and stripped
    - [X] Line codes: Bi-phi-M/S and delay modulation (IRIG-106)
    - [ ] Line codes: 4B5B, 8b/10b
- [ ] [Link](https://github.com/Anonoei/RPPS/tree/main/src/rpps/link)
    - [X] CRC, checksums, composite checks
    - [X] HDLC (streaming, any block cut)
    - [X] Headers: fields, a CRC over them or a frame check sequence over header and payload, header-driven lengths
    - [X] Byte-stuffed framing: SLIP, KISS, asynchronous PPP; checksums (sum, Internet, Fletcher, Adler, XOR); composite checks
    - [X] Fields as definitions: conditional, repeated, expression-sized, signed, scaled, enums, text (AX.25 addresses, BLE AD structures, TS adaptation fields)
    - [X] GSE (PDUs in DVB-S2 data fields)
    - [ ] MPEG-TS packets, CCSDS transfer frames and space packets, FX.25
- [ ] [Profiles](https://github.com/Anonoei/RPPS/tree/main/config/profile)
    - [X] DVB-S, DVB-S2 (normal and short frames, APSK), IEEE 802.11b, IEEE 802.15.4, Bluetooth LE advertising, Mode S / ADS-B, AX.25 AFSK, CCSDS telemetry (Reed-Solomon)
    - [X] CCSDS concatenated (convolutional + Reed-Solomon)
    - [X] LTE downlink to SIB1 (`lte/downlink`: MIB, CFI, DCI 1A, PDSCH, branches driving one another)
    - [X] 5G NR downlink to SIB1 at 15 kHz, through LEO satellite Doppler (`nr/downlink`: SS/PBCH, MIB, CORESET 0, DCI 1_0, PDSCH)
    - [X] DVB-S2 multiple input streams (`dvbs2/mis`)
    - [X] Bursts (`{"sync": "burst"}`, preambles) received and transmitted (placed, ramped, hopping); branches driving one another
    - [X] P25 phase 1 control channel (`p25/tsdu`), DMR bursts (`dmr/burst`)
    - [X] MF-TDMA by burst time plan (`rpps.plan`: DVB-RCS2's model, from a definition, the forward link's tables, or learned); `dvbrcs2/forward`
    - [X] Params with choices, variants, `when` on steps, children patching single steps
    - [X] Decoded messages (`Message`), only the kinds wanted, what each profile needs (`requires`), `profile.Decoder` from a selected band
    - [ ] From definitions already shipped: CCSDS turbo, IEEE 802.11a/g, 802.11n, ATSC 8VSB, GPS C/A, DAB
- [X] [Parallel](https://github.com/Anonoei/RPPS/tree/main/src/rpps/parallel): a profile's per-frame work on threads or processes, batch kernels
    - [X] GPU (CUDA or ROCm/HIP through CuPy): LDPC, BCH, demapping, the DVB-S2 region end to end
    - [ ] FPGA backend
- [X] [Analysis](https://github.com/Anonoei/RPPS/tree/main/src/rpps/analysis): blind survey and identification of unknown signals
    - [X] Signals in a capture (`survey`, `find_signals` read by level, a streaming `Finder`, `measure_band`), symbol rate, modulation (PSK/QAM/APSK, FSK index, CPM/SOQPSK offline), OFDM and its pilots, TDMA bursts, their kinds and their time plan
    - [X] Codes (hard or soft): convolutional (mother code, puncturing, RSC and turbo, TCM), linear block, Reed-Solomon, polar, quasi-cyclic LDPC (generator and sparse checks); scramblers, interleavers, frame markers
    - [X] CRCs: generator, init, xorout and coverage from frames (`framing.crc`); header fields (`framing.fields`); randomness (NIST SP 800-22)
    - [X] Step by step through a receive chain to a profile, or what is still unsolved (`analysis.profile`, `rpps solve`)
- [X] [Viz](https://github.com/Anonoei/RPPS/tree/main/src/rpps/viz): time, frequency, eye, constellation, spectrogram, survey, bursts and plans

## Not yet built

Reviewed 2026-11-19 against errorcorrectionzoo.org's realized codes, AFF3CT, liquid-dsp and gr-satellites.

- **Codes:** NR polar parity-check bits (an uplink message of 12-19 bits), LDPC limited-buffer rate matching from a
  transport block's TBS_LBRM, HARQ combining across redundancy versions; LDPC tables for CCSDS AR4JA / C2, DVB-S2X, DVB-T2,
  ATSC 3.0; UMTS and DVB-RCS duo-binary turbo; 4B5B, 8b/10b; fountain codes.
- **Waves and sync:** coherent CPM (GMSK, SOQPSK) in the data path (offline only: `analysis.modulation.cpm`); VSB / PAM, chirp
  spread spectrum, space-time codes; a burst step in a parallel region; DVB-RCS (first generation) tables over MPEG-TS sections, DVB-RCS2's duo-binary turbo code and reference waveforms; a preamble `offset`
  with known timing `anchor` (the grid lands off); the transmit side of branches (and of `grid`:
  `grid.lte.tx`, `grid.nr.tx` are test transmitters); NR at 30 kHz and above, NR VRB-to-PRB interleaving, several
  code blocks a transport block, multiple antenna ports; LTE beyond SIB1 (unicast DCI formats, multiple code blocks, TDD, extended prefix); `lte/downlink` in
  real time at 15.36 MS/s (about 1.1x slower; `nr/downlink` keeps up at 7.68 MS/s); a first-class carrier estimator for
  staggered signals, fractionally spaced block equalization, DFE; the DVB-S sync bytes, 802.11b SFD and DVB-S2 SOF as
  `sequence/word`; matched filter fused with Oerder & Meyr.
- **Links, sources, profiles:** MPEG-TS, CCSDS TM / AOS frames and space packets, FX.25,
  block-coded headers; tun/tap, audio devices, transmitting to a radio; profiles for DMR voice, P25 phase 2 and voice, 802.11a/g/n, ATSC, GPS C/A,
  DAB, UMTS, CCSDS turbo (and `projects/` using the BLE, ADS-B and 802.11b profiles that exist); `dvbs2/normal` dummy frames;
  positions and gaps in the parallel runner's output; MPower and Oerder & Meyr on a GPU; an FPGA backend.
