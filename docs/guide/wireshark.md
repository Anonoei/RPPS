# Packets: Wireshark and interfaces

A profile's messages (every frame it decodes: fields, payload, whether its check passed) can be watched live in
Wireshark, each as a packet of the link type Wireshark dissects it as, with what RPPS knows of it alongside: the
profile, the message kind, the check and its verdict, the frequency, the quality, the fields. Packets go the other way
too: written to an interface (or a capture), they are sent through the profile's transmitter.

```
rpps rx ble/adv capture.cf32 --fs 4e6 --pcap - | wireshark -k -i -          # live, from a file
rpps rx ble/adv --device hackrf --cf 2.402e9 --fs 4e6 --pcap - | wireshark -k -i -    # live, from a radio
rpps rx ax25/afsk1200 audio.cf32 --fs 48e3 --pcap pipe:rpps                 # Windows: wireshark -k -i \\.\pipe\rpps
rpps rx ieee802154/frames zigbee.cf32 --fs 8e6 --tun rpps0:ieee802_15_4_nofcs   # Linux: onto an interface
rpps tx ieee802154/frames --from-tun rpps1:ieee802_15_4_nofcs --device hackrf --cf 2.405e9 --rate 250e3
```

--8<-- "examples/wireshark_packets.py"

## Capture rules

A profile says how each message kind becomes a packet in `"capture"` (the rules and their transforms:
[`rpps.link.capture`](../api/link.md)); the link types are definitions, `config/linktype` (`python -m rpps ls linktype`).
The shipped profiles' rules:

| Profile | Kind | Link type | What is done |
|---|---|---|---|
| `ble/adv` | `pdu` | `ble_ll_phdr` (256) | access address before; the pseudo-header: RF channel, dewhitened, CRC checked and valid |
| `ax25/afsk1200` | `frame` | `ax25` (3) | the frame without flags or FCS |
| `ieee802154/frames` | `phr` | `ieee802_15_4` (195) | the PSDU with its FCS, which Wireshark checks |
| `ieee80211b/long` | `plcp` | `ieee802_11_radiotap` (127) | radiotap: rate (from SIGNAL), channel, FCS at the end |
| `dvbs/*` | `packet` | `mpeg2_ts` (243) | transport stream packets |
| `dvbs2/ts` | `ts` | `mpeg2_ts` (243) | user packets cut across BBFRAMEs (SYNCD, UPL), CRC-8 checked, sync byte put back |
| `ccsds/tm_packets` | `space` | exported PDU, `ccsds` | space packets cut across transfer frames per virtual channel (first header pointer) |
| `dvbs2/*`, `dvbrcs2/forward` | `bbheader` | exported PDU, `dvb-s2_modeadapt` | the BBFRAME, header and data field: Wireshark reads GSE and what it carries |
| `lte/downlink`, `nr/downlink` | `pbch`, `pdsch` | exported PDU, `lte_rrc.*` / `nr-rrc.*` | the MIB, the SIB1 transport block |

A kind without a rule goes out as Wireshark's exported PDU to its `data` dissector (Mode S, P25, DMR), its columns
saying the profile, kind and fields. A rule chooses the frame's parts (`include`: header, payload, FCS), restores what a
link type expects (bit order, a prefix, a sync byte, PPP's compressed address and protocol), adds a pseudo-header,
drops idle frames (`drop`), and names the fields for the comment. A standard framing Wireshark knows (Cisco HDLC, PPP)
is captured whole, with its link type, and Wireshark separates its header; `"linktype": "find"` finds where a known
link type starts after a header nobody describes.

## In Wireshark

Each (profile, kind) is an interface of its own (`rpps:<profile>:<kind>`). Filters:

- `frame.comment contains "FAILED"`: frames whose check failed (also `frame.packet_flags_crc_error == 1`).
- `frame.interface_name == "rpps:ble/adv:pdu"`.
- With `--pcap-rpps` (or `PcapSink(..., rpps=True)`) and `tools/wireshark/rpps.lua`
  (`wireshark -X lua_script:tools/wireshark/rpps.lua`, or copy it to Wireshark's plugins folder), each packet carries an
  RPPS header shown as fields: `rpps.crc_ok == 0`, `rpps.profile == "ble/adv"`, `rpps.field == "type.name=ADV_IND"`,
  before its own dissector.
- `--pcap-clock wall` stamps packets with the time they are written instead of the stream's own.

## Interfaces (Linux)

`rp.source.Tun` is a TUN or TAP interface both ways ([`rpps.source`](../api/source.md)). A TUN is raw IP, or on Linux
another link type it is set to (802.11, 802.15.4 without FCS, AX.25 with its KISS byte); a TAP is Ethernet. A packet
of a link type that is the interface's with transforms is converted (a link type's `"from"` in config/linktype: 802.15.4
with its FCS checked and taken off, AX.25 given its KISS byte), and back when transmitting; other packets are skipped
and counted: the pcapng stream carries any. Make the interface once, owned by you, and
RPPS opens it without root:

```
sudo ip tuntap add dev rpps0 mode tun user $USER
sudo ip link set rpps0 up          # (after RPPS set its link type: it is set while the interface is down)
```

Windows and macOS have the pcapng stream (a named pipe, a FIFO, stdout) instead.

## Transmitting

`rp.profile.Encoder` undoes a capture rule (a header part gives the frame's fields; an FCS is made again) and sends each
packet as one frame or burst; `rp.source.DeviceSink` transmits with a pysdrlib radio (`rpps tx --device`).

```python
# docs: skip
enc = rp.profile.Encoder("ieee802154/frames", rate=250e3, linktype="ieee802_15_4_nofcs") # (Linux's: no FCS, made here)
with rp.source.Tun("rpps1", linktype="ieee802_15_4_nofcs") as tun, \
     rp.source.DeviceSink("hackrf", cf=2.405e9, Fs=8e6, tx_gain=0.2) as radio:
    for pkt in tun.packets():
        enc.packet(pkt) >> radio
```
