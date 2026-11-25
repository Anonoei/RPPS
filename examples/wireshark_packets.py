"""Decoded packets in Wireshark, and packets back out: a profile's capture rules, pcapng, and the transmitter

    python examples/wireshark_packets.py                          # writes and checks rpps_packets.pcapng (in the temp dir)
    python examples/wireshark_packets.py - | wireshark -k -i -    # the same, live in Wireshark

A profile's messages (each decoded frame: its fields, payload and check) become packets of the link type Wireshark
dissects them as, by the profile's "capture" rules (rpps.link.capture): Bluetooth LE with its pseudo-header (channel,
CRC checked and valid), AX.25, 802.15.4 with its FCS. Each (profile, kind) is an interface of its own in the capture,
and each packet's comment says what it was. Here: a Bluetooth LE advertisement and two AX.25 frames are made with the
profiles' own transmitters, received, written as pcapng (to a file, or stdout with "-"), read back and checked; then
the BLE packet goes back out through rp.profile.Encoder (the capture rule undone) and is received to the same bytes.
"""
import sys
import tempfile
from pathlib import Path

import numpy as np

import rpps as rp

def receive(p, x, Fs, sps):
    rx = p.rx()
    rx(rp.Data(x, Fs=Fs, sps=sps, flush=True))
    return rx.take_messages()

def main(target=None):
    target = target or str(Path(tempfile.gettempdir())/"rpps_packets.pcapng")
    # Bluetooth LE: an ADV_NONCONN_IND carrying the advertiser's address and its name
    ble = rp.profile.load("ble/adv")
    adv = bytes.fromhex("112233445566") + bytes([2, 1, 6, 5, 9]) + b"RPPS"
    o = ble.tx()([rp.Data(adv, Fs=1e6, sps=1)], fields={"type": 2, "txadd": 1}, flush=True)
    ble_msgs = receive(ble, np.asarray(o.data), o.Fs, 4)

    # AX.25 (APRS): two UI frames over 1200 baud AFSK
    ax = rp.profile.load("ax25/afsk1200", centre=0, sps=10)
    call = lambda c, ssid, last: bytes(ord(ch) << 1 for ch in c.ljust(6)) + bytes([0x60 | ssid << 1 | last])
    frames = [call("APRS", 0, 0) + call("N0CALL", 7, 1) + b"\x03\xf0" + text for text in (b">hello", b">from rpps")]
    tx, s, waves = ax.tx(), None, []
    for i, f in enumerate(frames):
        d = rp.Data(f, Fs=1200.0, sps=1, stream=s, flush=i == len(frames) - 1)
        s = d.stream
        waves.append(tx(d).data)
    x = np.concatenate([np.zeros(400, np.complex64)] + waves + [np.zeros(400, np.complex64)])
    ax_msgs = receive(ax, x, 12e3, 10)

    # Every message as a packet of its link type: written as pcapng
    packets = ble.capture.packets(ble_msgs) + ax.capture.packets(ax_msgs)
    with rp.source.PcapSink(target) as sink:
        packets >> sink
    if target == "-":
        return
    recs = list(rp.source.Pcap(target))
    assert [r.dlt for r in recs] == [256, 3, 3] # LINKTYPE_BLUETOOTH_LE_LL_WITH_PHDR, LINKTYPE_AX25
    assert recs[0].data[10:14] == bytes.fromhex("d6be898e") and adv in recs[0].data # access address, then the PDU
    assert [r.data for r in recs[1:]] == frames # AX.25 frames from the destination address, no FCS
    assert recs[0].comment.startswith("ble/adv pdu, ok (crc24_ble)") and "type.name=ADV_NONCONN_IND" in recs[0].comment
    print(f"{target}: {sink.count} packets, interfaces {sink.per}")

    # And back out: the BLE packet through the transmitter, received to the same bytes
    enc = rp.profile.Encoder(ble, rate=1e6)
    y = np.concatenate((enc.packet(recs[0].data).data, enc.flush().data))
    again = ble.capture.packets(receive(ble, y, enc.Fs, 4))
    assert [k.data for k in again] == [recs[0].data]
    print("the BLE packet, sent again and received: the same bytes")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
