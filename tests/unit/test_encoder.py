"""Transmitting packets (rpps.profile.Encoder): a capture rule undone, the frame sent, received and captured again to the
same bytes; and out of a radio (rpps.source.DeviceSink), here a stand-in for a pysdrlib device"""
import numpy as np
import pytest

import rpps as rp
from rpps.errors import ConfigError
from rpps.link.capture import exported_pdu, unexported_pdu

def received(p, x, Fs, sps):
    rx = p.rx()
    rx(rp.Data(x, Fs=Fs, sps=sps, flush=True))
    return rx.take_messages()

def test_ble_packets_round_trip():
    """A packet as the capture gives it (pseudo-header, access address, PDU, CRC) in, the same packet out"""
    p = rp.profile.load("ble/adv")
    adv = [bytes.fromhex("112233445566") + bytes([2, 1, 6, 5, 9]) + name for name in (b"RPPS", b"AXES")]
    enc = rp.profile.Encoder(p, rate=1e6)
    # The packets a receiver captured: from the profile's own transmitter first
    first = received(p, np.concatenate([np.asarray(enc.bits(a, fields={"type": 2, "txadd": 1}).data) for a in adv]
                                       + [np.asarray(enc.flush().data)]), enc.Fs, 4)
    pk = [p.capture.packet(m).data for m in first]
    assert len(pk) == 2
    enc = rp.profile.Encoder(p, rate=1e6)
    y = np.concatenate([np.asarray(enc.packet(k).data) for k in pk] + [np.asarray(enc.flush().data)])
    again = received(p, y, enc.Fs, 4)
    assert [p.capture.packet(m).data for m in again] == pk and enc.count == 2 and enc.kind == "pdu"
    assert [m.fields["type.name"] for m in again] == ["ADV_NONCONN_IND"]*2 and [m.payload for m in again] == adv

def test_a_packets_header_gives_the_frames_fields():
    p = rp.profile.load("demo/hdlc_header", centre=0, sps=10)
    enc = rp.profile.Encoder(p, rate=1200.0, gap=0.02)
    assert enc.kind is None and enc.rule()[0].include == ["header", "payload"] and enc.kind == "hdr"
    pkts = [exported_pdu(bytes([9, 3, 0x40 + i]) + f"packet {i}".encode(), "data") for i in range(3)]
    x = np.concatenate([np.zeros(400, np.complex64)] + [np.asarray(enc.packet(k).data) for k in pkts])
    msgs = [m for m in received(p, x, enc.Fs, 10) if m.kind == "hdr"]
    assert [m.fields for m in msgs] == [{"address": 9, "control": 3, "protocol": 0x40 + i} for i in range(3)]
    assert [unexported_pdu(p.capture.packet(m).data)[1] for m in msgs] == [unexported_pdu(k)[1] for k in pkts]

def test_which_kind():
    with pytest.raises(ConfigError, match="no message kind"):
        rp.profile.Encoder("ax25/afsk1200", rate=1200.0, kind="nope").rule()
    with pytest.raises(ConfigError, match="not captured"):
        rp.profile.Encoder("demo/hdlc_header", rate=1200.0, kind="frame").rule()
    with pytest.raises(ConfigError, match="no burst step"):
        rp.profile.Encoder("ax25/afsk1200", rate=1200.0, hop=[1e3])

class FakeRadio:
    """What DeviceSink uses of a pysdrlib device"""
    CAN_TRANSMIT, NAME = True, "fake"
    def __init__(self):
        self.state, self.written, self.calls = {"open": False}, [], []
        self.freq, self.rate = 0.0, 1e6
    def open(self, **kw):
        self.state["open"] = True
    def close(self):
        self.state["open"] = False
        self.calls.append("close")
    def set_tx_freq(self, f, channel=0):
        self.freq = f
    def set_tx_sample_rate(self, Fs, channel=0):
        self.rate = Fs
    def set_tx_gain(self, g, channel=0):
        self.calls.append(("gain", g))
    def get_tx_freq(self, channel=0):
        return self.freq
    def get_tx_sample_rate(self, channel=0):
        return self.rate
    def start_tx(self, channels=None):
        self.calls.append(("start", channels))
    def stop_tx(self):
        self.calls.append("stop")
    def write(self, x, timeout=None, channel=None):
        self.written.append(np.array(x))
        return len(x)

def test_device_sink():
    radio = FakeRadio()
    with rp.source.DeviceSink(radio, cf=433.92e6, Fs=12e3, tx_gain=0.2, scale=0.5) as sink:
        assert (sink.cf, sink.Fs) == (433.92e6, 12e3)
        rp.Data(np.ones(100, np.complex64), Fs=12e3) >> sink
        [rp.Data(np.ones(10, np.complex64), Fs=12e3)]*2 >> sink
        with pytest.raises(ValueError, match="resample"):
            sink.write(rp.Data(np.ones(10, np.complex64), Fs=48e3))
    assert sink.count == 120 and np.allclose(np.concatenate(radio.written), 0.5)
    assert radio.calls == [("gain", 0.2), ("start", [0]), "stop", "close"] # Opened here, so closed here

def test_a_receiver_cannot_be_a_sink():
    class Receiver(FakeRadio):
        CAN_TRANSMIT = False
    with pytest.raises(ValueError, match="can't transmit"):
        rp.source.DeviceSink(Receiver())

def test_80215_4_frames_without_their_fcs():
    """Frames off a Linux 802.15.4 interface have no FCS: the Encoder makes it (config/linktype "from")"""
    import struct
    p = rp.profile.load("ieee802154/frames", sync={"timing": {"method": "known"}})
    mac = bytes.fromhex("41c8013412ffff0100") + b"no fcs here"
    enc = rp.profile.Encoder(p, rate=250e3, linktype="ieee802_15_4_nofcs")
    y = np.concatenate((np.zeros(256, np.complex64), enc.packet(mac).data, np.zeros(256, np.complex64)))
    (m,) = received(p, y, enc.Fs, 4)
    fcs = struct.pack("<H", rp.link.load("crc/crc16_kermit").encoder.compute(mac))
    assert m.payload == mac + fcs and p.capture.packet(m).data == mac + fcs
