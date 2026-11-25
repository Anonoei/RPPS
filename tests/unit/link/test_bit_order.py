"""Bytes on air in the order the standards send them: BLE, IEEE 802.11 and 802.15.4 send each octet least significant bit
first (a header's `payload_order`, the `interleave/block/lsb_first` definition). Expected bits are built from the
standards' rules, not from RPPS's transmitters, so a receiver and transmitter that agree with each other but not with
the air would fail here"""
import numpy as np

import rpps as rp

def air(octets):
    """Octets as the standards send them: each least significant bit first"""
    return np.unpackbits(np.frombuffer(bytes(octets), np.uint8), bitorder="little").astype(bool)

def test_ble_adv_pdu_bits_before_whitening():
    """Core 6.0 Vol 6 Part B 1.2: every field and each octet LSB first; 2.3: the 16-bit header (PDU type in the low 4 bits
    of the first octet, TxAdd bit 6, the length the second octet), the payload, then CRC-24 (init 0x555555) over both,
    sent as the catalogue appends it (CRC-24/BLE: reflected, little-endian)"""
    h = rp.link.load("header/ble_adv")
    payload = bytes.fromhex("0102030405060a0b") # AdvA 06:05:04:03:02:01 (LSB octet first), then an AD byte pair
    h.control(type=2, txadd=1)
    got = np.asarray((rp.Data(payload, flush=True) * h).data).astype(bool)
    pdu = bytes([2 | 1 << 6, len(payload)]) + payload
    crc = rp.link.load("crc/crc24_ble").encoder
    assert crc.compute(b"123456789") == 0xC25A56 # the reveng catalogue's check value
    assert np.array_equal(got, air(pdu + crc.compute(pdu).to_bytes(3, "little")))
    # Receive: the same bits give the payload back, as bytes most significant bit first
    h2 = rp.link.load("header/ble_adv")
    back = rp.Data(air(pdu + crc.compute(pdu).to_bytes(3, "little")), flush=True) / h2
    assert np.packbits(np.asarray(back.data).astype(np.uint8)).tobytes() == payload

def test_80211b_psdu_after_the_plcp_header():
    """802.11-2020 16.2.2 / 16.2.3: the PLCP fields and the PSDU's octets go LSB first"""
    h = rp.link.load("header/ieee80211b_plcp")
    psdu = bytes.fromhex("08014a")
    got = np.asarray((rp.Data(psdu, flush=True) * h).data).astype(bool)
    assert np.array_equal(got[-8*len(psdu):], air(psdu))
    rx = rp.link.load("header/ieee80211b_plcp")
    back = rp.Data(got[len(h.preamble):], flush=True) / rx # (SYNC and SFD: a delimit step takes them off)
    assert np.packbits(np.asarray(back.data).astype(np.uint8)).tobytes() == psdu

def test_80211b_psdu_any_block_cut(rng):
    """No FCS: the PSDU comes out byte by byte as it arrives, however the bits are cut"""
    psdu = bytes(rng.integers(0, 256, 37).astype(np.uint8))
    tx = rp.link.load("header/ieee80211b_plcp")
    bits = np.asarray((rp.Data(psdu, flush=True) * tx).data).astype(bool)[len(tx.preamble):]
    h, ctx, out, at = rp.link.load("header/ieee80211b_plcp"), None, [], 0
    for n in (5, 1, 33, 7, 100, 3, len(bits)):
        blk = rp.Data(bits[at:at + n], flush=at + n >= len(bits), ctx=ctx) if ctx is not None else rp.Data(bits[at:at + n])
        o = blk / h
        ctx, at = o.ctx, at + n
        out.append(np.asarray(o.data).astype(bool))
        if at >= len(bits):
            break
    assert np.packbits(np.concatenate(out).astype(np.uint8)).tobytes() == psdu

def test_802154_low_nibble_first():
    """802.15.4-2020 12.2.3: each octet's low nibble is the first symbol, its LSB the symbol's b0. The SFD 0xA7 is
    sent as symbols 7 then 10"""
    p = rp.profile.load("ieee802154/2450")
    chips = rp.sequence.load("ieee802154_2450").chips
    tx = p.tx()
    spread = next(op for how, op in tx if type(op).__name__ == "Spread")
    lsb = next(op for how, op in tx if "lsb_first" in type(op).__name__)
    sym = np.asarray((rp.Data(bytes([0xA7]), flush=True) * lsb * spread).data)
    sym = np.stack((sym.real, sym.imag), axis=1).reshape(-1) # (pairs: even chips on I, odd on Q)
    n = chips.shape[1]
    assert [next(i for i in range(16) if np.array_equal(chips[i], sym[k*n:(k + 1)*n])) for k in range(2)] == [7, 10]
