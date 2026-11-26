# Changelog

RPPS is 0.x: a minor version may rename or remove things, and old names are not kept as aliases (except the stable API
in the Building on RPPS guide, which warns for one release). Every release lists what breaks under **Breaking**, with
what to use instead.

## Unreleased

### Breaking
- **`rpps.grid.nr` takes either subcarrier spacing** (15 or 30 kHz), so its pieces changed shape:
  - `NrGrid` gives one frame a subframe (was a slot): `16 + 960 + 14 x Fs / 15 kHz` items, and `nr.unpack(frame)` is
    `(head, G, B)`: the grid of every slot `(2^mu, 14, nfft)` and the SS/PBCH block's own `(4, 240)`. The PDCCH publishes
    `slot`, which the PDSCH needs (in `nr/downlink`'s controls).
  - In `nr.phy`, `cp_lengths`, `symbol_starts`, `grid`, `modulate` and `symbol_phase` take the numerology `mu`,
    `slot_len(nfft)` is a slot of either, `ssb_place(pattern, i_ssb)` replaces `ssb_place(pci, i_ssb)` and `SSB_SYMBOLS`,
    `pbch_scramble` takes the scrambling's `v` (0-7), `pdsch_res(c0bin, prbs, s, l, mapping, dmrs_pos, blocks)` takes
    the resource blocks and the SS/PBCH blocks' footprints, and `SCS` is `SPACINGS`.
  - The sample rate is the block's spacing x a multiple of 256 (128 was accepted and could not hold the block).
- **`rpps.grid.lte`**: `phy.pdsch_res(blocks, symbols, pci, sf, n_rb, ports, duplex)` takes each slot's resource blocks;
  `tbs_table` is in `phy` (was in `channels`); grid frames carry the duplex mode in their header.
- **A constellation file lists only labellings, not their turns**: `psk/qpsk`'s R090 / R180 / R270 maps and `psk/bpsk`'s
  R180 are gone from the files, and first maps drop their "R000" prefix. The code makes every turn and conjugate of the
  first map as before (`rpps.symbol.mapping.expand_maps`), with the same names, so `load("psk/qpsk", "R090")` is unchanged.
- **`rp.analysis.signal_profile.rotation` is `rp.sync.inversion.sense`**, returning the same (True inverted, False, None).
- **The DVB-S2 constellations are four, not fifteen.** `psk/dvbs2_qpsk` is `rp.symbol.load("psk/qpsk", "dvb-s2")` and
  `psk/dvbs2_8psk` is `rp.symbol.load("psk/8psk", "dvb-s2")`: named maps on QPSK and 8PSK (a map's `"rotate"` turns the
  points while it is used: DVB-S2 8PSK sits at k*45 degrees; a map's own `"tags"` list it in `rp.tagged`, as
  `psk/qpsk (DVB-S2)`). The rate names `apsk/dvbs2_16apsk_2_3` ...
  `dvbs2_32apsk_9_10` are `rp.symbol.load("apsk/dvbs2_16apsk", rate="2/3")`: a `rate` choice sets the radii (no more
  `gamma`, `gamma1`, `gamma2` params). The `dvbs2/modcod` table has `mapping` and `params` columns, the constellation
  step and `Constellation.control` take `params`, and the turned maps are in `Constellation.turned`, not `maps`.
- **The scramblers `feedthrough/fib` and `feedthrough/galois` are removed**: generic registers no standard or system
  uses. `v35`, `g3ruh` and `ieee80211b` are the shipped feedthrough scramblers; write a definition of your own for
  another polynomial.
- **A definition file no longer says its `"base"`: the folder it is in does** (`fec/block/`, `filter/fir/`,
  `plan/builder/`..., else the resource: `crc/`, `profile/<family>/`). A file that writes `"base"` is refused:
  `python -m rpps fmt <your config dir>` takes out every `"base"` its folder says (and names a file in the wrong one). Definitions made in code still give `"base"`, and `rp.save`, the importers
  and `Guess.save` leave it out of the files they write. The plan builder moved to
  `config/plan/builder/dvbrcs2/forward.json` (`rp.plan.builder("dvbrcs2/forward")` as before).
- **`hdlc.frames_at` and `bytestuff.frames_at` give `(start, payload, ok, fcs)`**: the FCS bytes as received are the
  fourth (b"" without one), for a message's whole frame. `deframe` still gives `(payload, ok)`.
- **`rpps.profile.steps`, `rpps.analysis.identify` and `rpps.analysis.signal_profile` are packages** (each split into
  private modules); every name they defined is where it was. Names they only imported are gone from them:
  `rpps.analysis.identify.Guess` (and `Hypothesis`, `Report`, `pick`) is `rpps.analysis.result.Guess`.
- **The per-sample sync loops are second class, in `rp.sync.loops`; the first-class stages estimate in blocks.**
  - `rp.sync.Costas` is `rp.sync.loops.Costas` (now with `points=`: decision directed on any constellation, and
    `from_bandwidth`). The first-class carrier follower is the new `rp.sync.phase.DDCarrier` (below). A profile's
    `carrier: costas` still builds the loop.
  - `rp.sync.time.MM`, `time.TLL`, `time.ted` and `rp.sync.pll` (`PILoop`, `loop_gains`) are `rp.sync.loops.MM`
    (Mueller & Muller, Gardner), `loops.TLL`, `loops.ted` and `loops.pll`. A profile's `timing: mm`, `gardner` and `tll`
    still build them. The shipped profiles and examples time with `rp.sync.time.OerderMeyr`: `ax25/afsk1200`, `p25/tsdu`
    and `dmr/burst` over 256-symbol windows, ungated, and they decode more than Gardner did at every noise level but the
    cleanest (P25 TSBKs 287/295, 214/266, 89/178, 15/58 of 300; DMR slot types 317/320, 294/316, 249/298, 204/256 of
    320; AX.25 packets 56/60, 53/59, 22/33 of 60; Gardner / Oerder & Meyr, 20 seeds each). The RDS examples and
    `projects/fm_radio` use `OerderMeyr` and `MPower(order=2)`. `viz.eye.sync_phase` takes OerderMeyr's state.
  - **`rp.sync.AGC` is block estimation**: `AGC(target, *, window=1e-4, min_samples, log, min_gain, max_gain, init)`,
    keywords only after `target`. Each window's level sets the gain at its centre, a line between centres: a level
    step settles within a window, with no tail. `mu` is gone (a profile's `{"sync": "agc"}` refuses it and takes
    `window`). The loop is `rp.sync.loops.AGC(target, mu)` (`{"sync": "agc", "method": "loop", "mu": ...}`).
  - The adaptive equalizers `rp.sync.equalize.LMS` and `equalize.CMA` are `rp.sync.loops.LMS` and `loops.CMA` (with
    the `equalize` kernel). A profile's `equalize: lms` and `cma` still build them; the first class is
    `rp.sync.equalize.BlockLS` (`blind=True` where decisions cannot open the eye).
  - The per-sample loops that `DCBlock`, `Deemphasis`, `AMDemod`, `Squelch` and `PeakAGC` replaced (below) are
    `rp.signal.loops.DCBlock`, `rp.signal.loops.Deemphasis`, `rp.wave.loops.AMDemod`, `rp.sync.loops.Squelch` and
    `rp.sync.loops.PeakAGC`, with their kernels; the same output to rounding.
- **`rp.analysis.find_signals` is a new detector** (`rp.analysis.spectrum.find`): thresholds set by a false-alarm rate
  (`pfa`: expected false boxes in a capture of noise), a noise floor fitted per bin, and each signal read by level
  (`rp.analysis.spectrum.levels`): a line raised from the floor in steps of a few noise deviations; a signal is what
  comes up and holds its sides, and one coming up out of a wider one stands on it (`over`; the wider `holds` it). Lines
  (tones, spurs) and anything under 3 dB SNR are not signals. Signals get time extents (`t0`, `t1`, `bursts`; one with
  bursts is a `Burst`) and `score`. Its arguments after `nfft` are `cf, pfa, edge, drop_db, min_bins, min_across,
  nested`: pass them by name. `min_snr_db`, `gap`, `occupied`, `min_band_snr_db`, `edges`, `edge_db` and `split_db` do
  nothing and warn for this release, then go. `edge` defaults to 0.005 (was 0.02), also in `measure_band`.
- **Burst analysis is `rp.analysis.burst`** (was `rp.analysis.spectrum.burst`), a package: `preamble` (what the bursts
  share), `demod` (each burst demodulated), `plan` (`units`, `learn_plan`, `check_plan`); every public name is also at
  the package level. `rp.analysis.spectrum.burst.demodulate` (stable API) warns for this release, then goes; the rest
  of the old module is gone, its `track` loop too (each burst's carrier is followed by block estimation).
- **`rp.analysis.catalogue.SYSTEMS` is gone**: the systems are `config/system` definitions, one file each (`extends`,
  `abstract`, tags and user layers work as for any definition); `catalogue.systems()` loads them. `match` reports
  systems the features can't tell apart as `tied` instead of ordering them by name, and gives no positive score on weak
  evidence: it needs two agreeing features, or one strong one (baud, chip_rate, scs, code_length, profile, subcarrier,
  or an unmodulated carrier).
- **`{"signal": "stage"}` is `{"signal": "class"}`**, and every category takes a `class` step: your own operator by module
  path (`"class"`, `"args"`), a Stage for signal, sync and wave (`"tx_class"` for transmit), a Pipe for symbol, code and
  link. A class whose CATEGORY is another category's is refused.
- **BLE, IEEE 802.11b and 802.15.4 payload bytes are in the standards' bit order** (each octet least significant bit
  first). `ble/adv` and `ieee80211b/long` sent and read payloads most significant bit first, so a real packet's payload
  came out bit-reversed per byte (its CRC still passed), and `ieee802154/2450` sent each octet's high nibble first. A
  link header's new `"payload_order": "lsb"` does it for `ble_adv` and `ieee80211b_plcp`; `interleave/block/lsb_first`
  is the last step of `ieee802154/2450`, whose receive output is now data bits (Rep.DATA), not channel bits.
- **`fec/block/trellis/p25_1_2` is one codeword per 96 bits** (its new `length`): encode and decode take whole
  codewords, each started and flushed on its own, so it works as a profile's block step. A whole stream decoded as one
  trellis sequence is `decode_batch` on a row.
- **Keys nothing reads are reported** (`rp.errors.ConfigWarning`): a definition's when it loads; a profile's own, every
  step's, its `defaults` and the `sync` given to `load`, once built. A misspelling or a setting the method doesn't take
  no longer passes silently; a user definition that relied on that now warns.
- **`rp.math.interp.doubled` is `rp.math.interp.upsampled(x, a, b, base, u)`**: any factor u dividing 32, `u=2` as before.
- **`rp.analysis.signal_profile.framed` gives a seventh item**, the evidence (`{"verified", "refuted", "checks"}`), and
  ranks by it.


### Added
- **`lte/downlink` receives far more cells** (`rpps.grid.lte`):
  - TDD as well as FDD: the SSS's place says which; TDD's DCI 1A, PHICH size (tried until a DCI checks) and
    synchronization signals. PDSCH in the special subframe 1 is not received.
  - DCI 1C, distributed virtual resource blocks (both gaps) and the extended PHICH duration; table `lte/tbs_1c`.
  - Several cells on a carrier: `rpps.grid.lte.search` lists them, each found one's synchronization signals taken out
    before the next is looked for, and the profile's `pci` (and `duplex`, `span`) says which to receive.
  - `projects/lte/lte_sib.py` is a front end to these: every cell on the carrier through SIB1 and the SI messages (it
    took 4-port FDD cells with a localized 1A); `lte_scan.py` reports TDD cells and the other PCIs on a carrier.
- **`nr/downlink` at 15 and 30 kHz** (`rpps.grid.nr`): the SS/PBCH block at `scs` (patterns A, B, C; 4 or 8 candidate
  blocks, told by the PBCH), control and SIB1 at the spacing the MIB names, so the four FR1 combinations; CORESET 0 of
  TS 38.213 Tables 13-1 to 13-6 (`min_bw` for the 40 MHz ones) and interleaved VRB-to-PRB mapping. A block whose k_SSB
  says it has no SIB1 is counted (`no_sib1`).

- **DVB service information: the NIT, the time (TDT/TOT) and the rest of SI, with the PSI.** `header/link/dvb_si`
  reads (and builds) a section of every PSI/SI table: PAT, CAT, PMT, NIT, SDT, BAT, EIT, TDT, TOT, RST, ST, the CRC-32
  checked where there is one (not the TDT; the TOT too), descriptors by tag (network and bouquet names, satellite,
  cable and terrestrial delivery systems, service list, service, short event, local time offset, CA, language, stream
  identifier), others as bytes. New profiles `dvbs2/si` and `dvbs/si` give a message per section on the PSI/SI PIDs
  (and the PMTs' in `pids`), its table in `table_id.name`; `dvbs2/sections` (abstract) is what `dvbs2/mpe` and
  `dvbs2/si` share. Example: `examples/dvb_si.py`, in the Wireshark guide.
  - Field schemas (`rpps.link.fields`) read `"type": "bcd"`, `"mjd"` (an ISO date) and `"time"` (`"hh:mm:ss"`), DVB
    strings (`"encoding": "dvb"`, by their table byte), `"bytes": "*"` (the rest of the group), a list filling a byte
    count (`"bytes"` with `"repeat": "*"`), and shared `"groups"` a field names (`"group"`); `Schema.build` takes a
    `scope`.
  - A link header's `fcs` can be `{"crc": ..., "if": <expression>}`: only frames whose fields say so carry it.
  - `rp.sync.Align(publish=True)` (an align step's `"publish"`) publishes each period as a frame start, so a header can
    follow DVB-S's transport packets; `dvbs/qpsk`'s align steps have ids, `packets` and `frames`.
  - A packets step's `defaults` take a list of numbers (`pid in more`).
- **What a carrier's frequency is worth.** The signal profile's `frequency` step gives each carrier and symbol rate
  with its standard error from the receive context (pysdrlib's `ppm_accuracy` and a chain's `lo_accuracy`), and says
  which is not known when one isn't. Measuring them is `projects/reference` (LTE cells: the radio's reference error and an
  LNB's LO, and the pysdrlib calls that keep them), not the library's.
- **A carrier's level.** `Signal.carrier` (its power without the band's noise, dBFS) and `Signal.cn0` (dB-Hz), in the
  survey's listing. The signal profile's `level` step gives C/N0 (from Es/N0 and the symbol rate once demodulated, and
  names a gap to the spectrum's), dBFS, the gains, and dBm at the antenna port when the receive context holds a level
  calibration (pysdrlib's `level_offset` / `level_accuracy`). Calibrating (the radio's floor and noise figure, a tone, a
  noise source) is `projects/reference/power.py`. Which feed each carrier of a two-feed capture is on (leakage,
  frequency reuse) is `projects/reference/polarization.py`; `projects/reference/reference.py` is now `frequency.py`.
- **Sources know how their samples were received.** A pysdrlib `RxContext` (pysdrlib 0.5.1: the chain in front of the
  radio and its LO, whether it inverts the spectrum, the reference correction and clock source, gains, antenna) is
  `stream.rx`: `Device` asks the radio, `SigMF` reads `pysdrlib:rx`, and `File` / `Memory` take `context=`. A chain
  whose LO is above the signal is undone at the source (conjugated, `correct=False` to keep it; `stream.rx_conjugated`),
  `SigMFSink` keeps the context (marked `undone`), and `Memory.capture` carries it. The signal profile adds a
  `frequency` step (the carrier over the air, with what its reference is worth) and says whether an inverted spectrum
  is upstream (the receive chain known) or could be our own path.
- **Acquisition names QAM and APSK, and measures the roll-off.** `rp.sync.detect`, `Lock` and `Acquire` try the PSK, QAM
  and APSK definitions (`acquire.GROUPS`; children such as `dvbs2_16apsk` aren't other modulations), not only PSK. With
  `betas=None` (now the default) the roll-off is also read off the spectrum's edges (`rp.sync.rate.find_rolloff`: the
  matched-filter likelihood can't tell 0.05 from 0.15), and its nearest standard one (`rate.ROLLOFFS`, DVB-S2X's 0.05 to
  0.35) joins the three always tried; `Detection.rolloff` holds the measure, and the signal profile shows it ("roll-off
  0.05 (edges 0.06)"). A lock rechecks only the search's two runners-up each block (`Lock.RECHECK`), so an update costs
  what it did with three candidates.
- **Spectral inversion is found and put right, not picked as a map.** `rp.sync.Inversion(known)` (profile step
  `{"sync": "inversion", "header": ...}` or a complex `sequence`) matches a known sequence that turns (DVB-S2's
  pi/2-BPSK SOF) forwards and backwards on the symbol steps, so before carrier and frame sync, decides the sense by
  accumulated evidence and conjugates the stream back. `dvbs2/normal` and `dvbs2/mis` have it after timing: an inverted
  DVB-S2 carrier decodes as is (the pool's 23.8 MBd group). `CodeDetection.inverted` says the sense by a code on a
  standard's labelling (a shipped profile's map that decodes only conjugated, or only as received; None when that can't
  tell), and a signal profile always has a `spectrum` step: inverted, normal, or not observable. Random data on a
  constellation the same conjugated (every PSK, QAM, APSK) can't show the sense: only known structure or a code can.
- **16QAM, 32QAM, 16APSK and 32APSK are definitions**: `qam/16qam` (Gray per axis, and the `3GPP` map LTE and NR
  use), `qam/32qam` (the cross, quasi-Gray), `apsk/16apsk` and `apsk/32apsk` (the DVB-S2 labelling, free radii
  `gamma` / `gamma1`, `gamma2`). `apsk/dvbs2_16apsk` and `dvbs2_32apsk` extend them, the `rate` setting the radii as
  before. `rp.symbol.load("16QAM")` now finds `qam/16qam`; `generate` still makes any shape.
- **The blind tools try turned constellation maps.** `Constellation.blind_maps` (`maps`, then each turn and inversion
  of the maps that turn the points, such as DVB-S2 8PSK on 8PSK) is what `rp.analysis.identify` searches; a code found
  on one gives a profile step naming that map, and a correction that turns the symbols onto its points.
  `FrameSync(classify=...)` scores each candidate's turned point sets behind its word (the phase is absolute there) and
  publishes `mapping` with `modulation`.
- **Only your definitions, or only some standards**: `RPPS_CONFIG_SHIPPED=0` / `rp.use_shipped(False)` searches only
  the directories of `RPPS_CONFIG` / `rp.add_config_path`, so nothing shipped loads or is tried;
  `RPPS_CONFIG_TAGS=dvb-s2,lte` / `rp.select_tags([...])` has the blind tools (survey, identify, solve, the catalogue)
  try only definitions carrying one of those tags, plus the untagged generic ones. `python -m rpps ls` says when either
  is on, `ls --blind` lists what is tried.
- **The `example` tag: definitions made up to show a feature load and list, but no blind tool tries them.**
  `identify`, the survey's framed profiles, `solve`, the system catalogue and the CRC, scrambler, convolutional code
  and PSK searches list with `blind=True` (new on `rp.profile.ls`, `rp.code.ls`/`names`, `rp.link.ls`,
  `rp.code.scram.ls`, `rp.symbol.ls`). The shipped demos (`profile/demo/*`, `plan/demo/mftdma`, `table/demo/*`,
  `header/link/demo_*`, `sequence/word/demo_uw*`) carry it, so `profile.candidates()` no longer tries the two demo
  profiles. A user layer's file decides how its name lists, over a shipped one.
- **A parent can hold its children as rows, `"variants"`** (`{"columns": ["name", ...], "rows": [...]}`, a dotted
  column a key inside a dict: `params.gamma`): each row loads, lists, tags and is extended as the child file it stands
  for, by the same name. The DVB-S2 BCH codes (`bch/dvbs2_normal`, `dvbs2_short`), the CCSDS turbo rates, the DVB-S2
  16/32APSK radii and the DVB-S punctured-rate profiles are rows now: 41 files fewer, every name as before (the APSK
  radii since moved to a `rate` param: see Breaking).

**Packets: Wireshark, interfaces, transmitting**
- **Decoded frames live in Wireshark.** A profile's `"capture"` rules (`rpps.link.capture`) make each message kind a
  packet of its link type (`config/linktype`, `python -m rpps ls linktype`), taking the frame's parts it names and
  restoring what the link type expects: Bluetooth LE with its pseudo-header (channel, CRC checked and valid), AX.25,
  802.15.4 with its FCS, 802.11b with radiotap (rate, channel, FCS), DVB-S transport packets, DVB-S2 BBFRAMEs
  (Wireshark reads GSE inside), LTE and NR MIB and SIB1 (Wireshark's RRC dissectors); other kinds as exported PDU data.
  Each was checked against tshark 4.7's dissectors (BLE and 802.15.4 CRCs, 802.11 FCS verified by Wireshark).
- **`rp.source.PcapSink`**: pcapng to a file, a FIFO, stdout or a Windows named pipe (`pipe:NAME`), live: an interface
  per profile and kind, a comment per packet (check and verdict, frequency, quality, fields), the CRC-error flag on a
  failed check. `rp.source.Pcap` reads pcapng and pcap back. `rpps rx --pcap TARGET` (`--pcap - | wireshark -k -i -`),
  and `rpps rx --device` receives from a radio.
- **`tools/wireshark/rpps.lua`**: with `PcapSink(..., rpps=True)` (`--pcap-rpps`) each packet carries RPPS's header,
  shown as fields to filter on (`rpps.crc_ok == 0`, `rpps.profile`, `rpps.field == "type.name=ADV_IND"`) before its
  link type's dissector.
- **`rp.source.Tun`** (Linux): TUN/TAP both ways. Raw IP, Ethernet, or a TUN set to another link type (802.11,
  802.15.4, AX.25: `TUNSETLINK`, with packet information so the kernel takes non-IP packets); a link type made from the
  interface's (`"from"`: 802.15.4 with its FCS) is converted. `rpps rx --tun NAME[:LINKTYPE]`, `--tap NAME`. Tested in
  a user network namespace (`unshare -rn`), where Wireshark sees `link/ieee802.15.4`.
- **Transmitting packets**: `rp.profile.Encoder` undoes a capture rule (a header part gives the frame's fields, an FCS
  is made again) and sends each packet as a frame or burst; `rp.source.DeviceSink` transmits with a pysdrlib radio.
  `rpps tx --from-tun / --from-tap / --from-pcap` and `--device --cf --gain`. A packet captured, sent and received again
  is the same bytes (BLE, AX.25, the demo below).
- **A header inside a frame**: `Frame` publishes each good frame's start and length and anchors it, and a link header
  reads packets, so `{"link": "frame"}` then `{"link": "header"}` strips and reads a frame's own header
  (`demo/hdlc_header`). Messages carry the whole frame as received and its parts (`Message.frame`, `parts`) and the
  check behind `crc_ok` (`Message.check`).
- **`ieee802154/frames`**: the 2450 PHY with its SHR found and taken off and the PHR read: each PSDU a message.
- **Finding a link type after an unknown header** (`"linktype": "find"`): offsets scored by the link types'
  signatures (IPv4 length and header checksum, an EtherType and the IPv4 inside it). Found behind 0-64 random bytes in
  every one of 1000 frames; random bytes match at score 2 never in 500, at score 1 in 0.8%.
- `config/crc/internet16` (the Internet checksum); expressions take `in` and `not in`.
- **Packets that cross frames** (`{"link": "packets"}`, `rpps.link.packets`): a stream of packets in fixed frames' data
  fields cut back out by a pointer and each packet's length, per channel with its frame count, a lost frame dropping only
  the packets that crossed it; a check a packet carries for the one before (DVB-S2's CRC-8 in the sync byte).
  `ccsds/tm_packets` (TM transfer frame header `header/link/ccsds_tm`, space packets by the first header pointer per
  virtual channel, idle packets and frames passed over; Wireshark's `ccsds` dissector reads them) and `dvbs2/ts` (TS
  mode user packets by SYNCD and UPL, the sync byte checked and put back, captured as MPEG-TS). A framed link header
  takes its frame `length` from the profile, and its `payload` expression sees it.
- **Transmitting packets that cross frames**: `dvbs2/ts` transmits a transport stream given back to back, packing it
  into TS-mode BBFRAMEs itself (SYNCD per frame, each sync byte the CRC-8 of the packet before), the same however the
  stream is cut. A packets step's `tx` names the fields it publishes for the header after it at each frame start.
- **DVB-S2 null-packet deletion** (EN 302 307-1 annex D.3): `dvbs2/ts` reads NPD (and ISSY) frames, taking the DNP
  byte off each user packet and putting the deleted null packets back, so the transport stream comes out as it went in.
  A packets step's `keep`, `trailer` and `insert`.
- **IP in transport streams**: `header/link/mpeg_ts` (the transport packet header, adaptation field and all),
  `dvbs2/mpe` (MPE datagram sections, `header/link/dvb_mpe`) and `dvbs2/ule` (ULE SNDUs, `header/link/ule`) of a `pid`:
  sections and SNDUs cut across transport packets by the pointer field, per PID with its continuity counter, CRC-32
  checked, the datagrams captured as raw IP (Wireshark reads them).
- A profile's controls can drive a parameter in one direction only (`{"calc": ..., "on": "tx"}`).
- Fixed: a DVB-S2 transmitter given a first block too short for a frame sent its PL header twice.
- Packets through a radio and back without hardware: pysdrlib's simulated radios share an air (pysdrlib 0.5.1,
  `pysdrlib.air`): `DeviceSink("sim", air="lab", transmit=True)` into one, `rp.source.Device("sim", air="lab")` out of it,
  in real time.
- The LTE and NR channels say which check their messages passed (`Message.check`: PBCH and PDCCH CRC-16 or
  CRC-24C, PDSCH CRC-24A, or NR's CRC-16 for small transport blocks). `DeviceSink` takes pysdrlib's sim made with
  `transmit=True` (it asks the device as opened, not its class).

**Sync and signal**
- **`rp.sync.phase.DDCarrier`**: follows a carrier from decisions on any constellation (APSK, QAM too) by block
  estimation tracked across 32-symbol windows, after a preamble or word has taken the carrier off; `decided.follow` for
  a whole array. A profile's `carrier: dd` (`constellation`, `window`). `demo/mftdma` decodes with it what Costas did at
  every SNR down to -1 dB, and on 16APSK with a drifting, noisy phase it never slipped in 40 runs.
- **`rp.sync.equalize.BlockLS(..., blind=True)`**: a window whose decisions fail the gate is solved by block least
  squares constant modulus, then decision directed once the eye is open. On a channel that closes the eye it opened all
  20 seeds of QPSK and 16QAM (decisions alone failed 8 and 20). A profile's `equalize: ls` takes `"blind": true`.
  `rp.sync.equalize.symmetry(points)`.
- **`timing: mat`, the matched filter and timing in one** (`rp.sync.time.MatTrack`): the filter evaluated only at the
  symbol instants, its energy maximum tracked across windows; follows a clock 20 ppm off on DVB-S2 like `oerder_meyr`.
  `"device": true` runs it a group of windows a launch on a GPU, as `FrameSync(device=True)` does frame sync's metric.
- **SNR before lock** (`rp.sync.quality`): `m2m4(y, points)` from the symbols' moments, `from_sequence(y, ref)` from a
  known sequence (CAZAC: any shift, phase and multipath), `cyclic(x, sps)` from the spectral correlation at the symbol
  rate (any noise colour outside the band), `blind(x, sps)` from the empty band's eigenvalues.
- **`detect(esn0=, min_evidence=)` and `Lock.find(esn0=, min_evidence=)`**: an answer only when sure
  (`Detection.evidence`, `Detection.over_noise`), else `mod` is None and `Lock.find` raises NoLock.
- **Frame sync by named words** (`rp.sync.FrameSync`, `{"sync": "frame"}`): alternatives (`sequences`: {name: word};
  `zc`: Zadoff-Chu roots and shifts), the one found published as `word` with its fields; a word mid-burst (`before`,
  `length`); real levels kept real (`polarity`, `scale`); `segments` for a carrier that turns across a long word (by
  default a profile correlates a word in 16-symbol parts, at most 4); each match's `scale`, `cfo`, `esn0` published with
  its frame, `classify` naming the constellation; `count=` frames to find. `controls` drive later steps from these
  fields. A frame step takes a word (`sequence`, or `word` + `bits`) instead of a header, mapped through the profile's
  own steps, so a unique word sent after the FEC is found under any rotation.
- **Words of any width**: a `sequence` word may be longer than 64 bits; `rp.sync.Delimiter` matches it by FFT
  correlation. `rp.sync.word_errors(n_bits, false_alarm, positions)`: the most bit errors a delimit step may allow.
- **Bursts on many carriers** (`rp.sync.Bursts`; `{"sync": "burst", "carriers": [...] | "plan": ...}`): each carrier
  channelized from a capture at any rate, its neighbours filtered out, cut at absolute positions; with a plan each burst
  is placed (superframe, frame, slot, tx type, assignment), the plan's origin acquired and tracked from the unique words,
  `gate: "plan"` cutting slots energy can't see. A burst's fields drive later steps and ride on its messages
  (`fields["burst"]`). `rp.sync.PlanTx` places bursts by a plan on transmit.
- `{"sync": "timing", "method": "oerder_meyr", "clock": false}`: the phase only (a burst a few windows long).
- **`rp.signal.FFTBank`**: many channels of a capture from one FFT a hop, each at its own width, rate and exact centre,
  the same output however the input is cut.
- `rp.math.runs(mask)` and `rp.signal.filters.correlate(x, t)` (a template slid along a signal, by FFT blocks when long).
- `rp.sequence.zadoff_chu(u, N, shift)`.

**Codes, links and definitions**
- **Codes bit by bit**: any block code takes `append` (check bits after the codeword), `extend` (an overall parity
  bit), `layout` (where each codeword bit is sent) and `xor` (a pattern on the sent word), decoded in reverse, soft or
  hard. A `linear` code takes its parity bits as equations; the new `expr` type gives each codeword bit as an expression
  of the message bits. `fsm` trellis codes take a `length`. New: `bch/p25_nid` (BCH (63,16) and its DUID bit).
- **`{"symbol": "marker"}`** (known symbols before each frame: a sequence, or a Zadoff-Chu root and shift a control
  sets per frame) and **`{"symbol": "puncture"}`** (symbols at fixed places dropped, inserted on transmit: P25's status
  symbols).
- **Fields as definitions** (`rpps.link.fields`): signed, scaled, per-field `order`, `enum` names, `bytes` and `text`,
  sizes as expressions, `"if"`, `"from"`, groups (`"bytes"` sized, `"align"`), `"repeat"` and `"until"`, a field's
  `"tx"` on build. Link headers take them; headers and frames take `"payload_fields"`. `frame/ax25` reads addresses,
  control and PID; `header/ble_adv` reads PDU types, AdvA and AD structures; `dvbs/qpsk` reads the TS adaptation field.
- **Checksums** (`config/crc` `"type": "checksum"`: sum, Internet, Fletcher, Adler, XOR; `rp.link.checksum`) and
  **composite checks** (`"type": "composite"`: CRC parts over transformed views of the message, concatenated or XORed;
  `rp.link.check.register`), wherever a CRC goes.
- **Byte-stuffed framing** (`config/frame` `"type": "bytes"`): `frame/slip`, `frame/kiss`, `frame/ppp_async`, any
  block cut.
- **GSE** (`{"link": "gse"}`, `rp.link.gse.pack`): PDUs in DVB-S2 data fields, fragmented and reassembled.
- **The DVB-S2 PLHEADER reads DVB-S2X MODCODs**: `dvbs2_pl`'s first field `s2x`; a header definition can turn a field's
  code words (`"turn"`), and a table can apply only where fields have given values (`"only"`).
- **Bit kernels** (`rpps.math.bits.distance`, `find`, `unstuff`; C++ in `bits.cpp`): Hamming distance to a word at every
  position, bit destuffing over many spans. HDLC, `scan_hdlc` (6x) and FrameSync's hard-bit search use them.

**Profiles**
- **A branching profile pipelined**: `blocks.pipelined(profile.rx())` runs the trunk in one thread and the branches in
  another (`BranchRunner.pipelined_parts`), with the same output and messages as one thread at any block cut.
- **`p25/tsdu` and `dmr/burst`**: P25 phase 1 control channel TSDUs (NID and TSBKs, CRC-checked) and DMR bursts (every
  slot type; the data bursts' BPTC payloads), with their definitions.
- **Burst time plans** (`rpps.plan`, a `plan` resource): TDMA / MF-TDMA plans in DVB-RCS2's model, with frequency
  hopping (`hop`: an expression, or `ble_csa1`) and trains. From a definition, from a forward link
  (`rp.plan.builder("dvbrcs2/forward")`, read by the new `dvbrcs2/forward` profile), or learned from bursts. A builder is
  a definition too (`"base": "builder"`, `rpps.plan.air`).
- `demo/mftdma` (three carriers, two rates, three burst types, LDPC units), `examples/mftdma.py`,
  `examples/zc_modes.py`, `examples/profile_variants.py`.
- **Branches that read part of a frame**: `take` (item ranges of each frame) and `select` (only frames whose fields
  an expression accepts); frames of any length where a trunk control drives the branch's `frame`.
- **Params take `"choices"`** (each optionally a bundle of fields), **`"type"`, `"min"`, `"max"`**, checked at load.
  `rp.profile.variants(name)` lists a profile's combinations; `rp.profile.info` shows them.
- **`"when"` on a list item** (a step, any definition's list) and string literals, `ceil`, `floor`, `round`, `log2` in
  expressions; `and` / `or` short-circuit.
- **A child profile patches single steps**: `"steps"` as a dict by step id (merge, `null` to remove, `"+before:<id>"`,
  `"+after:<id>"`, `"+start"`, `"+end"`).
- **`requires` narrows a search**: `rp.profile.fits(name, baud=, bw=, Fs=, seconds=, input=)` says why a signal can't be
  that profile, `rp.profile.candidates(...)` lists the variants that may. `requires` is checked at load.
- The `mpower` carrier step takes a generated `"shape"`.
- A region's frame whose header CRC fails is dropped (`crc_ok` False) instead of failing the batch.

**Analysis**
- **`rp.analysis.spectrum.find`**: `find`, `fit_floor`, and `Finder`, a streaming finder for displays (`f.step(block)`
  gives a Survey with ids kept, independent of how the stream is cut, about 5 ms an update at 8192 bins). A signal not
  found again stays with `stale=True` (and `seen`) for `hold` seconds (-1, the default: for good). The detector runs in
  C++ when built (`find_detect`). `rp.analysis.Burst`: a `Signal` on part of the time (`duty`, `last`, `period`).
- **Burst analysis from real captures** (`rp.analysis.burst`): weak, slow, long and framed bursts; bursts that touch cut
  apart (`framing.frames.split`); known symbols anywhere or none (`shared_segments`, `Bursts.blind`); each burst's
  carrier from its preamble, tone pair or spectral centroid; long bursts follow their symbol clock (`retime`) and give
  LLRs; the TDMA frame from slot clustering. `learn_plan` learns a plan from detected bursts, `check_plan` judges one
  against them, `units(lengths)` splits burst lengths into overhead and code-block units.
- **Burst views** (`rp.viz`): `bursts` (a plan over a spectrogram, or a raster of superframes), `burst_timing`,
  `burst_stack`, `burst_terminals`, `plan_evidence`.
- **Blind CRC search** (`rp.analysis.framing.crc.search`): generator, init, xorout and coverage from frames.
- **What each header bit is** (`rp.analysis.framing.fields.map`): constants, counters, terminal ids, lengths, enums,
  saved as a link header definition.
- **Block codes from soft values** (`rp.analysis.code.block.linear.soft_dual`, `soft_find_length`, `soft_score`):
  BCH(63,51) named blind to 5 dB Eb/N0 (hard bits: above 6). `linear.frame_units`: a block code laid out in each frame's
  payload, found from few frames.
- **`rp.analysis.framing.randomness`**: NIST SP 800-22 tests and byte entropy, to tell encrypted from compressed or
  structured data; a code carrying random-looking bits says so (`fec.payload`).
- **`identify` sees a payload that repeats** (`repeat_period`, `CodeDetection.repeat`).
- **`rp.analysis.profile`**: `progress=`, `jobs=` (signals in forked processes), `hint=` (what an earlier look found),
  `symbols=` / `max_blocks` (reads on until a signal has symbols enough); TDMA bursts demodulated burst by burst; survey
  boxes refitted to their signals, carriers a notch separates or that sit side by side profiled each on their own; the
  carrier offset of framed profiles from their preamble; a profile's variants tried (`profile_params`).
- **Catalogue**: SCPC, DVB-RCS/RCS2 bursts, DVB-S2X, an unmodulated carrier, PM TT&C; features `burst`, `profile`,
  `sync`, `ssb_period`, `subcarrier`, `subcarrier_baud`. `python -m rpps ls system`.

**GPU and tools**
- **Oerder & Meyr and MPower on a GPU** (`OerderMeyr(device=True)`, `MPower(device=True)`; a profile's `"device": true`
  on `timing: oerder_meyr` and `carrier: mpower`): each window's sums, symbols and correction on the device, the
  tracking the host's, the host's output to float32 rounding. Each window waits on the device, so with 1 ms windows
  they are slower than the C++ host paths on an integrated GPU (the DVB-S2 receiver 1.3x slower on a Radeon 860M).
- **Device benchmarks re-run on an idle machine** (Radeon 860M against 16 CPU threads, 16-64 frames a batch): LDPC
  1.9-2.4x, BCH 2.0-3.6x, demap 4.2-8.6x, the DVB-S2 region 2.1x, the receiver end to end 1.1x; `timing: mat` on the
  device makes the receiver 16% faster. `tools/parallel_bench.py --front-device` puts the front's timing and carrier on
  the GPU (`--device` is the region alone, with any timing).
- **`tools/lte_bench.py`**: `lte/downlink`'s real-time factor in one thread and pipelined, and each step's load.
- **A parallel region on a GPU, CUDA or ROCm/HIP alike** (`Parallel("thread", xp=xp)`): LDPC, BCH and soft demapping
  run device kernels, with permutation, identity and header kernels so the DVB-S2 region stays on the device; bit for bit
  with the serial runner. On a Radeon 860M against 16 CPU threads the DVB-S2 region is 1.9x faster. `tools/device_bench.py`;
  Contributing, "Device kernels (GPU)".
- **Benchmarks keep results per machine** (`--save`, `--update-baseline`, `--check`, `--compare`; `.benchmarks/`).
  New: `tools/finder_bench.py`, `tools/modclass_bench.py` (the Es/N0 modulation detection works from),
  `tools/blind_bench.py` (what the blind code tools break, from how many symbols), `tools/ident_bench.py` (how far
  `rp.analysis.profile` gets blind on each shipped profile's signal, placed in a wider capture: box, rate, code, name).
- **Frames behind a sync marker, found blind** (`rp.analysis.framing.marked.layer`): a shipped profile's marker at one
  period, each additive scrambler restarted per frame, and the Reed-Solomon code inside at its interleaving depth
  (words that decode within t, against a random word's chance). `identify` runs it on channel bits and behind a Viterbi,
  so `ccsds/tm_rs` and `ccsds/tm_concat` are named from their symbols, at any depth.
- **`rp.profile.lineage(name)`**: the profiles a profile extends, nearest first.
- **`tools/ident_bench.py` variants and seeds**: `--variants plain inverted pad neighbour drift`, `--seeds`, the stop
  reasons tallied; DVB-S2 multistream, CCSDS TM packets, P25 and DMR signals.
- **`rp.symbol.mapping.bit_rotations`**: a constellation's turns that act on its labels as bit swaps and inversions
  (Gray QPSK's 90 degrees); **`rp.analysis.framing.frames.grid_of`**: the turns a constellation is symmetric under;
  **`rp.sync.acquire.onset`**: where a signal comes on in a block.
- **`rp.sync.FrameSync(cfo="track")`**: the carrier frequency tracked across frames from each frame's known symbols
  (the preamble and the header word it matched), started again when it changes, and refined by the phase from one
  preamble to the next; what it leaves (`cfo_left`) drives `rp.sync.PilotPhase(left=)`, which takes it off before its
  pilot line. The DVB-S2 profiles use it: with a 4e-4 or 2e-3 Rs carrier offset QPSK 1/2 decodes from 10 down to 2 dB
  Es/N0 (it decoded nothing at 6 dB and below: the 8th power never locks there, and the pilots alone alias).
- **One signal the survey cut into several boxes is profiled as one** (`rp.analysis.profile`): adjacent boxes whose
  cyclic spectra show one rate far under their widths (a spreading code's period: 802.11b's two halves either side of
  its Barker dip, 802.15.4's four pieces), or that are OFDM on one subcarrier grid and one symbol clock (pieces between
  empty subcarriers), are taken together; carriers side by side, each filling its own box or keeping its own clock,
  stay apart. The bench's `box` stage grades the box profiled.
- **Signals the linear chain doesn't take are named by the shipped profile that decodes them**
  (`rp.analysis.profile`): an unnamed signal is tried against each fixed-rate profile its box fits, on that profile's
  own channel of the capture (centred on the band the box sits in, filtered and resampled to its rate; an FM audio
  input on the discriminator's output), and named when its own checks verify, none refuted. GFSK bursts (`ble/adv`),
  pulse position (`adsb/mode_s`) and OFDM (`lte/downlink`, by its cell search and PBCH) are named from a capture;
  each profile's receiver does the work, nothing per standard in the analysis.
- **A drifting carrier is measured and taken off** (`rp.sync.acquire.drift`, `Detection.drift` in Hz/s): the M-th
  power's frequency segment by segment (the line through them), then its phase block by block (a quadratic), each
  only when it stands out of its own scatter. `detect` scores the symbols with the drift off, so Es/N0 under a Doppler
  rate reads as without it (1 MHz/s on QPSK: 3.8 dB low before, now within 0.1 dB); the profile shows it and takes it
  off the symbols its framed profiles read. The ident bench's drift variant: Es/N0 within 0.4 dB on the linear
  profiles (it read up to 16 dB low), all six named.
- **A good lock no code fits is read again** (`rp.analysis.profile`): the search's runner-up constellation, then twice
  and half the rate within the channel, the one that finds a code or a name kept (and said: a `retry` step); how many
  is `rpps._effort`'s `retries` (fast none, balanced 2, max 4). **`Hypothesis.p`** and **`Report.p(h)`**: a
  hypothesis's p-value, alone and over every hypothesis tried (the z_crit threshold tests at 0.01).
- **A profile built on the one a code names is named when its own layer verifies** (`rp.analysis.profile`): a packet
  layer's check is its packets' lengths meeting the frames' pointers (`rp.link.packets` counts `aligned`), so
  `ccsds/tm_packets` is named for space packets and `ccsds/tm_rs` for frames of other data.


### Changed
- **Faster where a receive found or rebuilt structure.** `rp.sync.Align` scores every bit offset, polarity and pattern
  position at once (`rp.math.bits.periodic_scores`): an acquisition attempt 2 ms instead of 165 ms,
  so DVB-S keeps up (4.1x to 0.67x real time in `tools/profile_bench.py`). A burst profile builds the steps after its
  burst step once and copies them per burst (BLE advertising 4.7x to 1.1x). A definition is expanded once and kept
  until a file of its `extends` chain changes, and config paths are resolved once.
- **Faster demapping and timing.** The LLRs share one exponential per point (the bit sums made from them), symbols go
  in cache-sized tiles, a long row splits over threads, and a row's noise comes from at most 4096 of its symbols
  (`rpps.math.demap.EM_SYMBOLS`, evenly strided, the same on the host, in numpy and on a device): one thread 2x (QPSK)
  to 4.6x (16APSK) faster. `OerderMeyr`'s window sums and the cubic interpolation (`interp.fractional`, 4 taps) run
  in C++: 62 to 157 Msamples/s.
- `tools/profile_bench.py`: each profile's receive on a synthetic signal, its steps' load and the heaviest step kinds.
- **Made-up codes, filters and windows are tagged `example`** (`ldpc/3_6`, `ldpc/gallager_96_48`, seven filters, three
  windows). Those built on a published method cite it in `ref`; `ldpc/3_6`'s comment no longer calls it a (3,6)-regular
  code, which it is not.
- **The tests are grouped by package**: `tests/unit/<package>/`, with `standards/`, `native/` and `policy/`, shared
  helpers in `tests/support/` (Contributing, Testing).
- **Definition files put on one line whatever fits in 120 columns** (`python -m rpps fmt`, `rp.save`): a profile's
  steps read one step a line (`{"sync": "carrier", "id": "carrier", "order": 8}`), `{"param": "sps"}` is one line, and
  config/ is a fifth shorter. Lists of numbers stay on one line; table `rows` one row a line.
- **A definition's `comment` can be a list of lines** (joined by spaces when it is read, at any depth); `fmt` cuts a
  comment too long for its line at a sentence or clause, so no line of prose runs past 120 columns.
- **A profile's position maps stay short on a long stream:** a change that keeps a step's rate (every header's) adds
  no anchor, and lookups no longer rebuild a list, so a receiver no longer slows down the longer it runs.
  `lte/downlink` at 15.36 MS/s went from 1.6x real time to 1.1x in one thread over 2 s of signal; the LTE grid and
  channel estimate also keep what a subframe's layout decides, and PDCCH decodes its candidates of a size in one
  Viterbi batch (`Viterbi.decode_batch` takes tail-biting codes).
- **No loop over samples or bits where a closed form or block form exists**, the same output to rounding at any block
  cut:
  - self-synchronizing scramblers are a FIR / recursion over GF(2) (`rp.math.gf2.fir`, `recur`): V.35 scrambles at
    195 Mb/s (was 0.7) and descrambles at 1.9 Gb/s; `Line("biphase_m" | "biphase_s" | "miller")` at 76 Mb/s;
  - `IIR` without the kernels is one state-space system filtered by chunks (13-39x faster); `DCBlock`, `Deemphasis`,
    `AMDemod` and `Squelch` are `IIR` sections (`rpps.signal.filters.iir.RunningMean`);
  - `rp.sync.PeakAGC` is a running maximum in logs (5-8x faster).
- **`rp.sync.detect` scores each candidate at its own Es/N0 from the symbols' moments** instead of letting each fit its
  own noise, which at low SNR named the densest candidate (QPSK at 4 dB was 64QAM 19 times in 20). `Detection.how`.
- **`rp.sync.Acquire` decides at fixed stream positions**, re-searching until two decisions agree: its detection no
  longer depends on how the stream is cut.
- **`AcquireFEC` decides at the end of a stream**, reuses the scans it already made as its window grows
  (`identify(prior=)`), and runs no blind tools on a rate Acquire only guessed (`min_line=5.0`).
- **Blind code identification**: soft decisions in the configured convolutional scan (K=7 1/2 down to Es/N0 2 dB, was
  6); convolutional before block codes, and the shortest block code of every bit map; the puncture search decodes a
  mother's patterns together (72 Viterbi calls where there were 1764); screens before full block and Viterbi scans
  (2.5-6x); the configured scan stops at an overwhelming fit (`decisive=20`); the blind Reed-Solomon search in C++
  (2.8x); `decode_tail` keeps nearly all of the window (DVB-S's RS(204,188) from 40k symbols, was 64k).
- **`frames.find` sees a short unique word from fewer frames** (40 where it took 80; `frames.comb_lags`);
  `frames.periods` gives `(lag, z)`. **`payload.fixed_word`** finds a word with bit errors (`word`, `bits`,
  `mismatches`, `errors`).
- **`matrix_mf` runs in C++** (9-10x). It and the numpy path raise IndexError when a symbol's support runs outside x,
  where numpy used to wrap round.
- `analysis.profile` channelizes every signal with one `FFTBank`; `signal_profile.framed` returns the profile's header
  ids as a sixth item and ranks a profile whose payload decoded first; `match_profile` returns (name, steps, params).
- `parallel_bench.py` and `resample_bench.py` take their positional arguments through argparse.
- **A framed profile is named by what its own checks verify.** `analysis.profile` ranks the profiles whose frames it
  finds by the steps whose check (CRC, FCS, a header's code) passed and those whose check kept failing, then the most
  general (`rp.profile.lineage`): DVB-S2 at 8 dB was named `dvbrcs2/forward` (the first by name, every DVB-S2 profile
  reading the same PL headers), a TS stream `dvbs2/ule`. A profile none of whose checks passes is not named.
  `identify` names the most general of the profiles with the same code steps (`ccsds/tm_rs`, not `ccsds/tm_packets`).
- **`rp.sync.phase.MPower` acquires from several windows** when one shows no M-th power line (`acquire`, 16384 symbols
  by default): DVB-S2 (M = 8) at 8 dB Es/N0 with a carrier 4e-4 of the rate off decoded no frame; now all but the
  first. Under 8 dB the pilots' phase still can't follow such an offset (a data-aided frequency is open).
- **`rp.sync.time.OerderMeyr` times from a sample a symbol** (it needed 2): under 4 samples a symbol its window is
  upsampled by the least power of 2 to 4 (`rp.math.interp.upsampled`) and the symbols read with the 12-tap sinc under 2.
  A 30 MBd carrier in 50 MS/s (1.67 samples a symbol) receives as at 3.33, with no re-extraction.
- **Blind code scans reach lower SNR.** `identify`'s Reed-Solomon scan scores words that decode within t when no phase
  has clean words (`rs_decode_scan`, off at effort "fast"): RS(204,188) behind a DVB-S Viterbi is found at 6 dB Es/N0
  (8 before). The blind convolutional scan weighs each parity relation by its bits' LLRs: an unknown K=6 code is found
  at 0 dB from 8192 symbols, where the hard count found none, its false alarms unchanged.
- **`rp.sync.Acquire` measures from where the signal comes on** (`onset`): 4000 samples of noise before a carrier read
  15 dB as 8.4 and lost its decode; now within 0.3 dB.
- **`analysis.profile` re-locks on a frame's unique word** when the continuous lock's decisions fail the gate
  (`framing.frames.pilot_frames`, frame by frame) instead of stopping there; frames are cut on their constellation's own
  symmetry (`grid_of`): BPSK was turned 45 degrees off its points.
- **`rp.sync.phase.MPower` tracks a carrier its windows can't show one at a time**: under the gate it holds the line and
  measures the last `acquire` symbols together (as it acquires); acquired again, it keeps to the 2 pi / M branch of the
  line it held. At 8 dB under M = 8 it acquired again and again, each time possibly 45 degrees round, mid-frame.


### Fixed
- LTE cell search: a Zadoff-Chu peak at an offset whole subcarriers away could outscore and hide the true one (seen
  with 1 and 4 transmit ports), so a cell was not acquired; every peak is now tried and the SSS decides. The DCI 1C size
  was counted over the wrong number of resource blocks (the same sizes by coincidence).
- **The `galois_right` register scrambles:** it shifted its feedback out before the output, so data went through
  unchanged after the first register length. It is now `galois_left` mirrored.
- **The `tukey` and `planck_taper` windows take odd lengths** (they raised `IndexError`).
- **`rpps gen-ldpc` writes definitions that load:** no `"base"`, and the shipped files' form (extending `dvbs2_64800`).
- **The `lowpass`/`highpass` filter prototypes name their shapes** when given an unknown one (`ValueError`, not
  `UnboundLocalError`).
- `Mat` could put a refined timing's first symbol's support before its buffer (numpy wrapped round to the block's end).
- `signal_profile.framed` failed on profiles that end in branches (`dvbs2/mis`), ending the framed pass for every
  profile.
- A profile with a guessed convolutional code failed to build (`Guess` read an old definition layout).
- `payload.signatures` found a periodic word in random bits (2 hits in 15 frames read as z 8) and called them structured.
- The blind convolutional scan found equal generators on any two equal bit streams (a BPSK carrier read as QPSK) and
  called the signal solved; pairs with a common factor are skipped.
- Which of two equivalent convolutional codes won the configured scan depended on the codes tried before.
- `analysis.profile` called a framed signal solved on its first headers alone, and a burst signal on demodulation
  alone; a frame with under 64 payload symbols raised and lost the signal's profile.
- `spectrum.burst.demodulate` found only one of two burst kinds on a carrier when the gate opened late on the first
  bursts.
- A profile's change for a step that had not run yet was placed as if positions started at 0 (every burst of a burst
  profile).
- `rp.sync.Burst`'s noise floor stopped at a handful of unusually quiet windows and called a whole capture one burst.
- `BurstTx` refused a burst that starts a little before the last one, within its `lag`.
- A block definition's `decoder` that can't be built from its code raised AttributeError; it is a ConfigError saying why.
- A framed link header with no payload gave no frames, and a header whose last bit ended a block was lost at the end
  of a stream.
- A profile's output joined decoded bits and passed-on LLRs as one array (the bits read as LLRs); it is all LLRs now.
- **The configured Reed-Solomon scan read dual-basis codes in the conventional basis** (`identify`'s `scan_rs`):
  CCSDS RS behind a Viterbi was never found.
- **`analysis.signal_profile.framed` lost every profile's frames when one candidate failed to build.**
- **`identify` names a profile at the interleaving depth it found** (`match_profile` binds the params a profile's code
  steps take: `ccsds/tm_rs` at depth 1 came out at its default, 5).
- **A signal at 2 samples a symbol locked at a wrong rate** (`rp.sync.rate.find_baud`, `Lock.find`): |x|^2 put its
  rate line on Fs/2, outside the search, and a noise peak was taken (16QAM at 0.17-0.53 of its rate). With no clear line,
  x is interpolated to twice its rate and searched to Fs/1.5, a line there taken when clear and the signal as wide as it.
- **`ble/adv` filters to its channel before the discriminator**: over 4 MHz of noise it fell under FM's threshold
  (15 dB Es/N0: 18 of 123 packets; now 116 of 153, every one at 20 dB).
- **A profile's checks verify it when they pass in a tenth of its frames or more** (`analysis.signal_profile`; was: as
  often as they fail): a CRC-24 passing on 28 of 149 weak packets is no chance. And a profile built on the one a code
  names isn't taken when a layer of its own gives nothing (`dvbs/si` on transport packets that carry no tables), nor a
  framed profile one of whose checks is refuted (DVB-S2's PLS read on 802.15.4 chips, its BBHEADER CRC never).
- **`rp.analysis.burst.shared_segments` raised a math domain error** on hundreds of bursts (a weak or absent network):
  the chance of them agreeing underflowed to 0. It is no shared segment, and the bursts are demodulated blind.
- **A convolutional profile decoded at only some carrier phases** (`dvbs/qpsk`: 5 of 12): nothing resolved the 90 degree
  ambiguity an M-th power carrier leaves. `rp.sync.Puncture` tries the constellation's turns (`bit_rotations`) with its
  offsets and applies the one that decodes; and it decides on the first window where one hypothesis clearly wins, not on
  the stream's first (where timing and carrier were still settling and every offset decoded badly).

## 0.6.0 (2026-11-22)

### Breaking
- **RPPS is licensed under the GPL, version 3 or later** (GPL-3.0-or-later), from this release; 0.5.0 and earlier stay
  MIT. A program that distributes RPPS, or a work based on it, does so under the GPL's terms.
- **`stream.dropped` counts source samples**, a radio's overruns and a threaded source's discarded blocks alike (it
  counted blocks for `Blocks.threaded(drop=True)`).
- `ldpc.min_sum(llr)` is `ldpc.propagate(llr)`: the layered schedule runs min-sum or sum-product (the definition's
  `"check"`, or the effort level's). `rpps._effort.iters(base)` is `scale("iters", base)`, `maxlog(base)` is
  `choice("maxlog", base)`.
- **A burst profile's `p.tx()` places the bursts:** a call gives the samples up to its burst's start (the gap before it),
  not the burst; `tx.flush()` gives the rest. Send the packets and write what comes out (`examples/profile_loopback.py`).
  `rpps tx --gap` is seconds, not samples.
- **pysdrlib 0.5 is required.** **Sample formats follow pysdrlib 0.5's names** (SigMF's: the bits of I alone): `cf32` is two float32 (numpy complex64,
  a `.cfile`), `cf64` complex128, `ci16t` int16 pairs (a gqrx `.cs16`), `ci8t` int8 pairs. What RPPS called `cf64` is
  `cf32`, `ci32t` is `ci16t`, `ci16t` is `ci8t`. `rpps rx/tx/survey/profile --fmt` defaults to `cf32`.
  Raw filter files (`import_fir`, `import_iir`, `read_table` `dtype=`) take the same names (`i8t ui8 i16t i32t` for the
  integers) or a numpy dtype with its byte order (`"<i2"`); a bare numpy code is refused (`"i8"` is int64 to numpy).
- `rpps.sync.acquire.refine_timing` is `rp.sync.track_timing(x, sps, beta, span=8, rounds=2)`.
- **The detection rides on each block:** stages read `meta.det` (and `Data.det`), the detection the block went through
  `Acquire` with, not `meta.stream.det`, which is only the latest. A stage of your own that read `meta.stream.det` should
  read `meta.det`; setting `stream.det` by hand still reaches blocks no Acquire stamped.
- **`ble/adv` receives packet by packet:** a burst step cuts the capture into packets and each is demodulated on its own
  (timing and carrier offset from its access address, found by a preamble step). Its runner gives a list, one output per
  packet, each with its own `stream.est`; `sync={"timing": {"method": "known"}}` is no longer needed.
- **Decode status comes with the decode; nothing of it stays on the code.** One loaded code can now decode on several
  threads (before, a thread could read another's status, and get its `failed` mask).
  - Through operators: `out = data / code`, then `out.errors` (per codeword: corrected bits or symbols, -1 where it
    failed). `Coding.errors` is gone.
  - Direct calls: `msg, errors = code.decoder.decode(x, status=True)` replaces reading `decoder.errors` after
    `decode(x)`. The same applies to `fsm.decode_batch(x, status=True)`.
  - CRC: `crc.ok` is gone; `crc.decode(bits, status=True)[1]` is `[0]` if it checked out, `[-1]` if not.
  - LDPC: `converged` is gone; it is `errors >= 0`, and `decode_codewords` returns `(hard, converged)`.
  - HDLC: `last` is gone; `decode(bits, status=True)` gives 0 or -1 per frame found, and `deframe(bits)` every frame
    with its FCS verdict and the aborts.
  - `Marker.errors` (the word's bit errors per frame) is on the decoded data: `out.errors`.
  - `Line.errors` is gone; a Manchester invalid pair is still marked in `out.failed`.
  - Turbo: `decoder.llr` is gone; `decoder.app(x)` gives the a-posteriori LLRs of the message bits (the decoded bits
    are `app(x) < 0`), as the BCJR decoders' `app` does.
  - fsm: the decided paths' output words come from `decoder.paths(x)` (bits, words), not from a private attribute.
  - A code implementation defines `_decode(x, ...) -> (output, errors or None)`; `_code.decode` wraps it.

### Added
- The `mm` timing step takes `biphase` (Manchester symbols, timed on their mid-symbol transition) and `interp`:
  RDS as a profile, built in code and run (`examples/e2e_profile.py`).
- **More of the stable API** (the Building on RPPS guide): `analysis.profile`, `source.Memory`, `waveform.classify`,
  `cumulants.classify`, `fsk.analyze`, `hopping.track`, `burst.demodulate` and `framing.frames` `find`, `cut`,
  `coherence`, `periods`: a change is called out, and a rename warns for a release.
- **`rp.sync.Lock.follow(x, at=pos)`:** the block right after a lock's, its symbols joining on with no symbol dropped
  or repeated at the seam, the carrier phase on the same branch of its M-fold symmetry; so a display gathering symbols
  over contiguous blocks locks only the new block (1.5 ms for 2500 samples, against 21 ms for `update` over 41000).
  `find` and `update` take `at` too; a block that isn't the next raises NoLock, and a timing jump over 0.3 symbol
  makes the lock stale.
- **Channels in shared memory:** `rp.source.ShmSink(name or [names])` publishes blocks (a channelizer's columns, one
  channel each) as pysdrlib shared-memory streams (`pysdrlib.shm.ShmStream`) at their positions, with each channel's
  centre, width, parent span and label for a view. Another process receives one like a radio,
  `rp.source.Device("shm@rpps.<name>")`. Example: `examples/channel_shm.py`.
- **`Device` follows a retune and ends when the device is lost:** a "freq" / "rate" event while streaming cuts the
  block there and starts the stream over at the new setting; pysdrlib's `err.Lost` ends the blocks (the last flushing,
  `device.lost` says why) instead of raising. `Device` also takes a target string (`"hackrf@SERIAL"`, `"shm@rpps.ch3"`,
  `"host/hackrf-0"`), opened with `pysdrlib.open` and closed by `close()`.
- **Load covers everything, and as one number:** `stream.load()` now times pipes (`/`, `*`: decoders) as well as
  stages, and a parallel region's workers; `stream.load(window=s)` over the last `s` seconds of signal;
  `stream.realtime(window=None)` the busiest thread's load (<= 1.0 keeps up).
- **GPU-ready parallel region, vendor neutral:** `rp.parallel.Parallel(..., xp=xp)` runs a region's kernels on an
  array module such as CuPy (CUDA or ROCm/HIP); `rpps._accel.device()` picks it (`RPPS_DEVICE=none|cuda|hip`). A plan
  uploads a batch once, downloads it once, and moves it explicitly around kernels that need numpy (`to:host`,
  `to:device` in `describe()`); kernel constants go to a device once (`Kernel.const`).
- **Positions stay true to time across a gap:** `Device` stamps each block from the radio's own count (pysdrlib's
  `index`), so samples it lost move `meta.pos` on, and every stage carries a gap in its input into its output positions
  (as it does a block `threaded(drop=True)` discarded). `stream.t0` is the radio's clock (`hw_time`) or the host's at
  the first sample, so `meta.time_of(k)` dates a live stream.
- **What an analyser shows of a lock:** `rp.sync.Lock` keeps its block (`lk.x`) and gives `decisions()`, `errors()`
  (per symbol: error vector, EVM and magnitude error as a percentage of the constellation's rms, phase error in degrees),
  `summary()` (EVM rms and peak, MER, magnitude and phase error, and from a fit of the symbols as kI I + kQ Q + c: IQ
  offset, gain imbalance and quadrature error), `derotated()` (x as the symbols see it), `matched(up)` (the matched
  filter between the symbols, `up` points a symbol: eye diagrams; one convolution read between the samples, its
  instants the symbols; `exact=True` the filter's matrix at every point) and `reference()` (the
  decided symbols through the ideal pulse). The measures are `rp.sync.quality.errors` and `summary` for any symbols.
- `Signal.channel()` gives the front end's `delay` [s] and `settle`: the output samples a new stream starts with before
  every filter is full (its start-up transient), so a display drops exactly those.
- **More effort knobs, measured** (`tools/effort_bench.py`: the lowest Es/N0 each chain decodes at, and its speed, per
  level). `max` decodes LDPC by layered sum-product (native `ldpc_sumproduct`: 0.19 dB over min-sum on DVB-S2 64800 1/2
  QPSK, 0.47 dB on 16200 3/5 8PSK, in fewer iterations), lists 4x the polar paths (SC codes list 4; 0.25-0.5 dB), and
  gives a turbo product code one more Chase bit (with 4x its iterations: 0.47 dB). `fast` halves the polar list and takes
  one Chase bit off. An LDPC definition may name `"check": "sumproduct"` itself.
- **Only the message kinds wanted:** `profile.kinds()`, `profile.rx(kinds=...)`, `Decoder(..., kinds=...)`,
  `ProfileInfo.kinds`, `rpps rx --kinds`: only those kinds come out, and only the steps they need run (their steps and the
  headers that drive them): unneeded branches are not built, a chain ends at its last needed step. LTE's MIB alone
  takes about half the time, DVB-S2's PL headers alone skip the LDPC (about 20x faster).
- **Decoded messages:** every frame a profile receives is also an `rp.profile.Message` (kind, fields, payload, crc_ok,
  sample, time, freq, quality), from `runner.take_messages()`, placed in the stream the runner was given. Link headers,
  HDLC, the DVB-S2 PL header, the LTE and NR channels emit them; DVB-S TS packets (with PID) and CCSDS transfer frames are
  cut by the runner (`"message"` on a step); failed frames come too, `crc_ok` False. `rpps rx --messages` prints them as
  JSON lines.
- **What a profile needs:** `"requires"` in every profile (baud, Fs, bw, min_samples, input, from_cf);
  `rp.profile.info(name, baud=, **params)` and `rp.profile.catalogue()` describe them (`ProfileInfo`).
- **`rp.profile.Decoder(name, Fs, offset, cf, baud, bw, **params)`:** a selected signal's samples to the profile's
  messages: shift, staged decimation and a fractional resampler to the profile's rate (an FM demodulator for AX.25's
  audio), the rate measured first where the standard leaves it open. `feed(samples, start=, time=)` takes blocks that need
  not follow on (an overlap is cut, a short gap filled, a long one starts over); `status` says "measuring", "searching",
  "locked" or "lost", why, and the load.
- `analysis.code.block.ldpc`: blind quasi-cyclic codes. `qc_generator` solves the generator's circulants from about
  k/z + 9 words (each word gives z equations; the reliable words solve, held-out ones check), `circulant_size` finds z,
  and `qc_checks` finds sparse checks in the generator itself: sums of two to four shifted parity blocks whose
  information dependences nearly cancel. An LDPC code's dense generator hides those from a word-level search unless the
  words are many.
- `Lock.update(..., recheck=True)` scores the other modulations the search tried on the same symbols and calls the
  lock stale when one fits better (a QPSK lock on 8PSK at 10 dB loses only some 5 dB of Es/N0).
- `rp.analysis.measure_band(x, Fs, freq, bw, cf)`: a band given by hand measured as `find_signals` measures what it
  finds (power, SNR, height, duty), as a `Signal`.
- `rp.wave.demod(..., kind="pm")`: the phase, carrier offset and phase removed, with peak and rms phase deviation.
- The stable API also holds `rrc_pulse`, `spectrum.cyclic` (`alpha_profile`, `features`, `ORDER_NAMES`) and
  `spectrum.scf.coherence`.
- `rp.sync.Lock`: a signal's symbols as a snapshot, for programs that seek, pause and retune. `Lock.find(x, Fs)` searches
  (raises `rp.errors.NoLock` with the reason), `lock.update(block)` follows rate, timing, cfo and phase on any later block,
  consecutive or not, with no search; `lock.stale` says when to search again; `lock.refine(longer)` searches the rate
  again on more signal. `Acquire` runs on it.
- `rp.sync.best_tau` runs the matched filter once and interpolates it (4-20x faster, the same timing); `best_tau`,
  `track_timing`, `matrix_mf` and `centres_in` are exported from `rp.sync`.
- `find_signals(nested=False)` leaves out bands that hold other signals; `Signal.id`, and `Survey.match(previous)` keeps
  ids from survey to survey.
- `rp.wave.demod(x, Fs, kind, bw)`: FM, AM, USB or LSB audio from one block, no state or delay, with FM offset and
  deviation, AM depth, and the tone and SINAD of a test tone.
- README: the API other programs can build on, and an import-cost test (no scipy, matplotlib or pysdrlib on `import rpps`).
- `analysis.spectrum.burst`: bursts of several kinds, each with its own preamble (a TDMA network's burst types each carry
  a unique word): when the bursts share fewer than `min_shared` symbols they are grouped by length, each group's own
  preamble found (`burst_kinds`), and every burst acquired on its kind's (`BurstResult.kind`, `Bursts.preambles`).
- **Speed: LTE and NR downlinks near real time.**
  - `polar_scl`, a C++ polar list decoder: about 20x the numpy one, the same paths and metrics. Polar's N is checked
    without building G.
  - NR's PDCCH decodes each aggregation level's candidates in one batch; the PDSCH's LDPC code is built once per size.
  - NR and LTE cell derotation from a two-table ramp, not an `exp` per sample.
  - LTE CRS channel estimation for every port and symbol in a few array operations; control-region maps cached.
  - Results unchanged: LTE outputs bit-identical on the band 13 capture.
  - NR: 3 s of a LEO pass in 2.3 s at 7.68 MS/s (was 22 s); live 2 s captures at 15.36 MS/s in about 2 s (was 15).
  - LTE: about real time at 15.36 MS/s (was 5x).
  - Cell search: the samples filtered and decimated once, each offset a turn (LTE) or a roll of their spectrum with
    half-size inverse FFTs (NR): LTE 470 ms -> 90 ms, NR +-60 kHz 520 ms -> 250 ms.
- **5G NR downlink to SIB1, through LEO Doppler (NTN):** the profile `nr/downlink` and `rpps.grid.nr` (15 kHz).
  - `NrCell`: PSS search over a wide carrier offset (default +-60 kHz), SSS, the block index and half frame from the PBCH
    DM-RS. Then every SS/PBCH block tracked by two second-order loops: the offset and its rate, the slot timing and the
    clock's rate. Out come whole slots from a frame start.
  - `NrGrid`: a frame per slot. `PBCH` (MIB and timing bits, polar), `PDCCH` (CORESET 0 and the Type0 search space from
    the MIB, CCE-to-REG interleaving, DCI 1_0 to the SI-RNTI), `PDSCH` (DM-RS, TBS, 16 and 64QAM, LDPC, the block's CRC).
  - `nr.tx`: a synthetic downlink, and `leo()`, a satellite pass (delay and Doppler from a circular orbit).
  - Tables `nr/coreset0_15_15`, `nr/type0_fr1`, `nr/pdsch_tdra_a`, `nr/mcs_pdsch_1`, `nr/tbs`. The SS/PBCH block, BCH and
    PBCH scrambling match py3gpp bit for bit (`tests/unit/vectors/nr_py3gpp_ssb.json`).
  - Through a 600 km pass at 2.185 GHz: every MIB and SIB1 over 3 s across closest approach, and at 46 and 50 kHz offsets.
  - On air (HackRF): two live cells at 15 kHz decode to SIB1, one in n5 (MIB 100/100, SIB1 41/42) and one in n25 (MIB
    100/100, SIB1 50/50).
  - The phase NR's upconversion gives each symbol (38.211 5.4): one cell applied it, one (sharing its carrier with LTE)
    did not. Given `fc` (the block's raster frequency), the cell search tells which (its offset measure is not fooled by
    it), and the PDSCH tries both, the CRC choosing.
- **DVB-S2 multiple input streams:** `dvbs2/mis`, one demodulator and decoder, then a branch per input stream. Branches:
  a control whose header is in the trunk drives steps in the branches (the PLS sets every stream's BBFRAME length). Link
  headers: `select` ({field: value}) keeps only the frames with those values (`skipped` counts the others).
  `dvbs2/normal` transmits multistream frames from each frame's `sis` and `isi` fields.
- **5G NR codes:** `nr_polar` (`block/polar/nr_pbch`, `block/polar/nr_dci`): the polar sequence and input interleaver of
  38.212, code size, puncturing and shortening, the sub-block and coded-bit interleavers, CRC-aided list decoding with
  the DCI's 24 ones and RNTI mask (`paths()` for a blind search). `nr_ldpc` (`block/ldpc/nr_bg1`, `nr_bg2`): both base
  graphs with every lifting set, a structured encoder, filler and the punctured 2 Z, the circular buffer at each
  redundancy version and the modulation-order interleaver; `segment()` and `base_graph()` for transport blocks. CRCs
  `crc24_nr_c`, `crc11_nr`, `crc6_nr`; tables `nr/polar_sequence`, `nr/polar_interleave`.
  Checked against py3gpp vectors (`tests/unit/vectors/nr_py3gpp.json`): LDPC and its rate matching, the CRCs and the
  polar encoder are bit-exact.
- **LTE downlink to SIB1:** the profile `lte/downlink` and the package `rpps.grid` (`grid` steps).
  - `grid.lte.LteCell`: cell search (PSS, SSS, carrier offset), then each half-frame's PSS tracked (the sample clock),
    the samples cut into whole half-frames from a subframe 0; acquired again when lost, zeros for what was missed.
  - `grid.lte.LteGrid`: a frame per subframe, its resource grid and the channel of ports 0-3 from the reference signals.
  - `grid.lte.PBCH`, `PCFICH`, `PDCCH`, `PDSCH`: the MIB and port count, the CFI, DCI 1A / 1C in the common search space,
    and the transport block a DCI 1A allocates (turbo, CRC-24A), with transmit diversity for 1, 2 or 4 ports. Each publishes
    what it decoded, so the profile's branches drive one another.
  - `grid.lte.tx`: a downlink (cell signals, MIB, CFI, a DCI and its PDSCH) to test with; the table `lte/tbs`.
  - `rpps rx --branch NAME`: a profile that branches gives that branch's bits.
- **Branches in profiles:** a last `{"branch": ...}` step fans the chain out into branches that read the same blocks
  (`rpps.profile.branch`), in order, each in its own stream; a header in one branch drives steps in the branches after
  it, at the trunk position of its fields; `frame` makes the branches read whole frames. Receive only.
- **Burst transmission in profiles:** `p.tx()` of a profile with a burst step is a `BurstTxRunner`: a call sends one
  burst, through steps built anew for it (no more `flush=True` a packet by hand), and gives the samples up to its start
  (`flush=True` or `tx.flush()`: the rest). The burst step places the bursts (`rp.sync.BurstTx`): `gap_s` apart, or at
  `at=` [s] (TDMA slots), with `ramp_s` raised-cosine power ramps, on a carrier each (`freq=`, or the step's `hop` list
  cycled) on a stream of rate `Fs`: MF-TDMA carriers may share a slot (they add), and a hopping carrier keeps the phase
  of one continuous oscillator. `tx.sent` records each burst's start, length and carrier. A preamble step with
  `"attach": true` sends its own waveform before each burst (`Preamble.attach_tx()`, a `rp.sync.PreambleTx`). `ble/adv` sends 1 ms apart.
  `rpps tx`: `--gap` is in seconds; `--hop`, `--rate-out`. Receiving MF-TDMA or hopping takes a channelizer before the
  profile (a channelized burst step is not built).
- **Burst demodulation in profiles:** a `{"sync": "burst"}` step; the steps after it are built anew for every burst
  (`rpps.profile.burst.BurstRunner`). A `{"sync": "preamble"}` step finds each burst's preamble, its waveform made by the
  profile's own transmit side; on real input (FSK) by the correlation coefficient, with the offset level taken off. Known
  timing takes an `anchor` (a preamble's find). `rpps rx` takes runners that give lists. Live BLE channel 37: every
  packet the hand-written receiver (projects/ble) gets, and 4 more of 106.
- `rp.sync.Burst`: a stream cut into the bursts that rise out of its noise floor, each a `Data` of its own stream at its
  absolute position, so what follows acquires every packet fresh. `Blocks` and `Pipeline` run what follows on each burst
  (a stage may now give a list). Its floor is taken from the noise windows alone (`quiet`), so bursts filling nearly
  all the air are found; zero windows (padding) count as the quietest noise.
- `rp.sync.Preamble`: a burst's preamble by a known waveform (normalized correlation) or its repetition (Schmidl & Cox),
  with the carrier offset the repetition gives, optionally taken off.
- `rp.sync.ToneLock`: an oscillator locked to a pilot, mixing at a harmonic (FM stereo's 38 kHz from 19 kHz; ATSC's pilot
  to DC); windows at absolute positions.
- A source checkout warns on `import rpps` when the in-place `rpps._native` was built from other C++ sources than
  `csrc/rpps_native` has now (`rpps._accel.stale`); rebuild with `python tools/builder.py -c`. Installed packages are not checked.

### Changed
- **`Survey.match` ids are a signal's own:** a signal missing for a survey or a few (`hold`, 8) gets its id back when it
  returns, and a new signal takes an id never used before in the chain, so signals coming and going never renumber the
  others (`survey.lost`, `survey.next_id` carry it on).
- `stream.busy` holds each operator's own time (what it runs inside it is counted for the inner operator, not twice)
  and includes pipes.
- `rp.source.Sink` leaves scaling to pysdrlib's `FileSink(scale=True)` (the inverse of `File`'s) instead of its own:
  the same bytes in every format.
- **Documentation in one place per kind, built with Zensical:** the site (https://anonoei.github.io/RPPS/) has guides
  (`docs/guide/`: the data model, receiving, profiles, definitions, unknown signals, real time, building on RPPS) that
  embed the tested `examples/` scripts, and an API reference rendered from the docstrings by mkdocstrings, replacing
  pdoc. The design that was README.dev.md is in the package docstrings (`rpps.base`: model, rule, categories,
  contracts, type tags, streaming rules; `rpps.sync`, `rpps.profile`, `rpps.parallel`, `rpps.analysis`, `rpps.wave`,
  `rpps.symbol`, `rpps.grid`, `rpps._effort`); its workflow parts are `docs/contributing.md`, the roadmap
  `docs/roadmap.md`, config/README.md `docs/guide/definitions.md`. README.dev.md, `src/rpps/README.md`, the
  subpackage READMEs, `examples/README.md` and `pdoc_templates/` are gone. The `docs` extra is zensical and
  mkdocstrings-python; `python tools/builder.py -d` builds into `site/`. `builder.py` moved to `tools/`.
- **For agents:** AGENTS.md, and `llms.txt` / `llms-full.txt` on the site (AGENTS.md, every
  package docstring and every public signature in one file), generated by `tools/gen_docs.py`.
- `examples/integrating.py`, shown in the Building on RPPS guide: a `Lock` followed across blocks, and a `Decoder` from a
  band at another rate and offset to `Message` records.
- Python blocks in README.md and `docs/` run in the tests (`tests/unit/test_docs.py`), as do checks that every example
  is shown and every package has a docstring.

### Fixed
- **`framing.frames.find` no longer frames noise:** one loud sample or a few positions coherent by chance passed as a
  unique word (65536 random QPSK symbols at 10 dB gave a "period" in 21 searches of 30, a different one each time).
  Products are clipped, and a word is taken only when chance could not have made it over every position and candidate
  period (`false_alarm`, 1e-3); with many frames the per-position threshold comes down, so real frames are found more
  often (13 frames at 10 dB, 0 dB over 65 frames), but never under `min_coherence` (0.2): a structure a few percent of
  frames share (pilot blocks at a short lag) is significant over thousands of frames, and not a unique word.
- **`find_signals` on a crowded band:** a signal a small gap from a louder neighbour was measured against the
  neighbour's level, not the gap's, and dropped out of some surveys; and the receiver's roll-off reaching just past
  the band edge was reported as a narrow signal with a negative SNR.
- `rp.source` refuses a pysdrlib older than 0.5 (ImportError naming the version and the fix): its format names
  differ ("cf32" was two float16), so a capture was read as noise and nothing locked.
- Eye diagrams (`rp.viz.eye`) are upright (they were drawn upside down), take a block with no whole trace (a stream's
  short last block), keep their alpha in range with few traces, and `eye.Q` updates with Q; a debug print is gone.
  `examples/e2e.py` and `e2e_mat.py` draw in rpps' dark style again (`rp.viz.subplots`, `rp.viz.styled()`).
- `analysis.spectrum.waveform.classify` judges the envelope of bursts inside them by the smoothed power
  (`waveform.on_samples`), not by each sample's: filtered PSK in bursts was called constant envelope, its dips between
  symbols dropped.
- `analysis.modulation.cumulants.classify` takes about 2 ms instead of 0.5 s: the moment-cumulant formula merges
  partitions with the same moments once (C84: 109 terms, not 4140), each candidate's theory is computed once, and the
  power share is fitted for every share at once.
- `analysis.spectrum.burst.demodulate` and `framing.frames.bursts`: a given `baud`, `mod` or `beta` narrows each
  burst's acquisition instead of being searched for again (the rate was searched on every burst even when given).
- `rp.sync.Lock.find`, `update` and `refine` raise `NoLock` with a reason for whatever holds no symbols, never numpy's
  errors: samples all zero ("no signal: the samples are all zero"), NaN or inf ("not all finite"), a rate of 0 when any
  line is taken (a constant), estimates that come out non-finite ("nothing like symbols here"). Their arithmetic runs
  under its own numpy error state, so a caller's `np.seterr(all="raise")` no longer turns an underflow into a
  `FloatingPointError`. `update` on zeros now says "all zero" (it said "fewer than 16 symbols").
- `rp.analysis.spectrum.scf.coherence` of silence (all-zero samples) is 0, not NaN with a 0/0 warning (an empty channel
  next to a signal already read 0); samples that are not finite are a `ValueError` saying so.
- `rp.analysis.spectrum.cyclic.features` likewise: silence has no lines whatever `np.seterr` says, samples that are not
  finite are a `ValueError` saying so, and a sequence varying only by rounding (an exactly constant input) no longer
  shows a symbol-rate line found in its residue.
- `rp.sync.BurstTx` (a burst profile's `p.tx()`): a burst added after `take(flush=True)` with an `at` before the
  flushed end failed inside numpy. A flush now ends the stream: the next burst starts a new one, its `at` from that
  stream's start (`sent` is still readable until then); a start in samples already handed out is refused by name.
- `rp.profile.Decoder.feed`: a step back of more than the block (a seek without `reset()`, file playback) was cut as an
  overlap, so every block was dropped until the index passed the old one; it now starts over. A long gap or a seek
  gives the frames the profile held instead of dropping them. `reset()` and a restart forget the block times and the
  last good message (messages took times from the old stream, `status` said "lost" from it).
- **Faster `Lock.update` and `wave.demod`** (a display following a signal every frame):
  - `best_tau` reads each candidate timing's symbol instants off the matched filter with a 12-tap windowed sinc instead
    of upsampling the whole output 16 times by FFT: `track_timing` about 5x faster (8 -> 1.5 ms a block of 8192 samples
    at 20 samples a symbol), and a followed rate no longer biased by the linear read between upsampled points (50007 ->
    50001 Bd on a 50 kBd signal).
  - `sync.candidates` reads each constellation once per config search path: a recheck no longer finds the files every block.
  - `wave.demod` skips the channel filter when `bw` is Fs and the audio filter when `audio_bw` is Fs/2 (the whole span),
    filters the audio on the half spectrum, and keeps complex64 input in single precision: FM over the whole span 20 -> 4 ms
    for 50000 samples.
- `analysis.spectrum.burst`: a long burst type (thousands of symbols, where the gate can open tens of symbols from the
  start) got no preamble of its own and was acquired on another kind's; `burst_kinds` now aligns with a near and a wider
  search. A burst far off carrier (a large share of a turn a symbol) was acquired on data; each burst's blind carrier
  offset now comes off before its preamble search, and the reported `cfo` includes it.
- `analysis.code.block.ldpc.qc_generator` refused words whose information bits include constants (a header's): it now
  solves with those unknowns free and lets the held-out words judge.
- An HDLC frame step flushed with fewer bits held than a flag raised instead of giving nothing.
- `analysis.code.block.linear.sparse_checks` returned nothing when given fewer than k + 16 words (all went to the solve,
  none were left to verify on). It now keeps some aside first.
- Pipelined (`.pipelined`, `.threaded`), a stage after `Acquire` could take a refinement made for a later block, its
  detection or its sps, depending on the threads' timing: the output differed from run to run.
- `sync.time.Mat` dropped a symbol when a refinement moved the timing a little earlier: it now moves to the nearest
  point of the refined lattice, not the first one after.
- `analysis.spectrum.waveform.on_off`: the period of bursts much shorter than their frame came out as the smallest lag
  searched (inside a burst the autocorrelation is near 1 at every lag). It is now the earliest true peak, within 10% of
  the highest, past the autocorrelation's first dip, per lag over its overlap, searched up to half the record (at most
  0.5 s). The power is smoothed over at least 32 samples (a narrow cut's symbols swing at 2 us).

## 0.5.0 (2026-11-20)

### Breaking
- **Loaders take one reference**, the path below the resource: `rp.code.load("conv/k7_171_133/3_4")`,
  `rp.link.load("crc/crc32_mpeg2")`, `rp.signal.filters.load("pulse/rrc/0p35")`. A loaded code is named by that path
  (`"rs/204_188"`), not its family folder.
- **`rp.symbol.load` reads definitions only** (`"psk/qpsk"`); shapes by name come from `rp.symbol.generate("16QAM")`.
  `load_or_generate` is for names a detector hands over.
- **Definition aliases are gone.** Definitions now have parents (`"extends"`, `"abstract"`) and tags
  (`rp.tags()`, `rp.tagged("dvb-s2")`, `rpps ls --tag`):
  - `constellation/dpsk` is gone: DPSK is a constellation after a differential step.
  - `sequence/recurrence/dvb_energy` is `sequence/recurrence/dvbs2_bb`.
  - Convolutional codes have one set of keys for encoder and decoder (the split form is gone).
  - Linear codes take `"n"`, `"k"` (not `"code": [n, k]`).
  - `profile/dvbs` takes the code rate as a parameter; the per-rate profiles pin it.
- **Profile steps name what they load:** `{"code": "conv" | "block" | "interleave", "name": ...}` instead of
  `{"code": "fec", "type": ..., "name": ...}`; a filter step is `{"signal": "filter", "name": "fir/..."}`; a puncture sync
  is `{"sync": "puncture", "conv": ...}`. Your own profiles need the same change.
- **`rpps.analysis` is a tree:** `spectrum/`, `modulation/` (`detect` was `modulation.py`), `sequence/`, `code/conv/`,
  `code/block/` (`linear` was `block.py`), `code/interleave` (was `interleaver.py`), `framing/` (`frames` was
  `framing.py`). No old paths are kept; the top-level helpers (`profile`, `detect`, `find_signals`, `survey`, `psd`) are
  unchanged.
- **Renamed in `rpps.analysis`:**
  - `spectrum.cyclic.spectrum_floor`
  - `sequence.recurrence.bit_windows`
  - `code.conv.tcm.label_windows`
  - `code.interleave.recurrence_z`
  - `spectrum.waveform.on_off`
  - `framing.telemetry.TelemetryFrame`
- **GF(2) helpers live in `rpps.math.gf2`:**
  - `berlekamp_massey` returns `(polynomial, length)`, plus `pdivmod`, `pmul`, `pgcd`, `porder`, `fwht` and `rank_ints`.
  - `code/matrix.py` is gone: use `math.gf2.parity_to_gh` / `gh_to_parity`.
  - `utils.inth.binary.entropy` is `math.bits.entropy`.
- **Removed:**
  - The real-time spectrum viewer (`viz/rt`, `viz/ot`, `viz/blit`, `viz/dialog`) and `analysis/stft` (use `analysis.psd`).
  - `base/buffer`, `source/err`, `convert.read_array`, `errors.DecodeError`.
  - `Signal.baud` (use `width`).
- **Kernels moved:** the kernel protocol (`Form`, `Batch`, `Kernel`) is `rpps.base.kernel`, and `parallel.kernels` is
  gone. Operator control (`publish`, `anchor`, `of`, `Control`) is `rpps.base.control`.
- `rp.viz` no longer sets a global matplotlib style on import.
- `tests/examples` moved to `examples/`, and `tests/dev` and `tests/prop` were removed.

### Added
- **Packet links from definitions:** line codes (NRZI, Manchester, bi-phase), sync markers, header FCS, and streaming
  HDLC. New profiles: BLE advertising, Mode S / ADS-B, AX.25 AFSK1200, CCSDS TM (RS and concatenated).
- **`rpps tx`:** a profile's transmission to a file. `rpps rx` and `rpps show` were also added.
- **Codes:**
  - LDPC with layered normalized min-sum.
  - Turbo product codes (Chase-Pyndiah).
  - Polar (SC, list, CRC-aided).
  - Reed-Muller, Reed-Solomon dual basis, and CCSDS turbo.
- **Sources:**
  - `Source.ppm`, which corrects Fs and cf for the radio's reference error.
  - `Source.REP` / `APPLIED`, and `profile.rx(source=src)`.
  - SigMF and WAV in, WAV out (`rp.source.write_wav`, `WavSink`), and TCP/UDP sources.
- **Signal processing:**
  - `rp.Tee`, which runs one stream through named branches.
  - `rp.signal.baseband`.
  - `sync.CPSync`, OFDM timing from the cyclic prefix.
  - TLL timing.
  - Effort levels (`rp.set_effort`).
- **Blind analysis:**
  - Cumulants, spectral coherence, matrix pencil, FSK / CPM / PM, and impairments.
  - Scramblers and linear recurrences.
  - Turbo constituents and interleavers, polar frozen sets, LDPC reconstruction, TCM checks.
  - Hopping, DSSS, telemetry framing, separation and cancellation, and the catalogue.
- **Faster:** native FIR / IIR / frequency shift, polyphase TX, a thread pool, cached code tables, and blind search in
  about a third of the time.
- `examples/`: one runnable script per recommended way of using RPPS, run by the tests.

### Changed
- `import rpps` no longer imports pysdrlib; File, Device, Sink and the formats import it when used.
- The data path no longer imports `rpps.analysis`.
- The TPC decoder follows Pyndiah: more iterations no longer hurt.

### Fixed
- SigMF sources unmap their data file on close, so Windows can delete it.

## 0.4.1 (2026-11-17)

### Changed
- Wider analysis library, modulations, FEC and scramblers.

## 0.4.0 (2026-11-13)

### Breaking
- Packages renamed: `serial` is `source`; filters, samples and channel are `signal`; `mod` is `symbol`; `coding` is `code`.
- `chain` is gone: a whole signal is a profile (`rp.profile.load`), with header-driven control.
- Sync estimators are block-rate loops, with per-sample loops as the fallback.
- Requires `pysdrlib>=0.4.3`.

### Added
- Optional C++ kernels (`rpps._native`, pybind11) with Python fallbacks.
- The frame-parallel runner, and batch / vectorized kernels.
- FEC definitions for P25, DMR, LTE, ATSC, BLE and Wi-Fi; a CRC of any width and length.
- Remez filter design, a chunked polyphase resampler, Oerder & Meyr estimation, and a least-squares equalizer.
- Signal quality, signal finding, the signal survey and its CLI.
- `projects/`: worked receivers on real signals.
