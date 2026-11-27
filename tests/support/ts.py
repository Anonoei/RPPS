"""MPEG transport streams for tests: packets of sections, mixed PIDs, DVB-S2 and DVB-S through their profiles, and
PSI/SI sections built byte by byte from ISO/IEC 13818-1 and EN 300 468 (not from the definitions under test)"""
import datetime
import struct

import numpy as np

import rpps as rp

NULL = bytes([0x47, 0x1F, 0xFF, 0x10]) + b"\xff"*184

def ts_stream(rng, n, pid=0x100):
    return [bytes([0x47, pid >> 8, pid & 255, 0x10 | (k & 15)]) + rng.bytes(184) for k in range(n)]

def ts_packets(units, pid, rng):
    """Units (sections, SNDUs) in the payloads of transport packets of one PID (ISO/IEC 13818-1 2.4.4.2): a pointer
    field where one starts, some packets with an adaptation field, 0xFF to fill the last"""
    stream = b"".join(units)
    starts = set(np.cumsum([0] + [len(u) for u in units[:-1]]).tolist())
    out, pos, cc = [], 0, 0
    while pos < len(stream):
        af = b"" if rng.random() < 0.7 else bytes([7, 0]) + b"\xff"*6
        room = 184 - len(af)
        first = min((s for s in starts if pos <= s < pos + room), default=None)
        if first is not None:
            room -= 1
            first = min((s for s in starts if pos <= s < pos + room), default=None)
        data = stream[pos:pos + room]
        payload = (bytes([first - pos]) if first is not None else b"") + data
        payload += b"\xff"*(184 - len(af) - len(payload))
        afc = 3 if af else 1
        out.append(bytes([0x47, (first is not None) << 6 | pid >> 8, pid & 255, afc << 4 | cc]) + af + payload)
        pos, cc = pos + len(data), (cc + 1) % 16
    return out

def mux(rng, *streams):
    """Transport packets of several PIDs, each in its order, mixed, with null packets"""
    queues = [list(s) for s in streams]
    out = []
    while any(queues):
        q = [q for q in queues if q][int(rng.integers(0, sum(1 for q in queues if q)))]
        out.append(q.pop(0))
        if rng.random() < 0.2:
            out.append(NULL)
    return out

def receive_ts(p, o, block=None):
    """(output, messages) of profile p receiving the samples o (in blocks of `block` samples, else at once)"""
    rx = p.rx()
    x = np.concatenate((o.data, np.zeros(200, np.complex64)))
    st = None
    for i in range(0, len(x), block or len(x)):
        d = rp.Data(x[i:i + (block or len(x))], Fs=o.Fs, sps=4, delay=o.delay, stream=st,
                    flush=i + (block or len(x)) >= len(x))
        st = d.stream
        out = rx(d)
    return out, rx.take_messages()

def through_dvbs2(p, ts, block=None):
    """Transport packets sent by dvbs2/ts and received by profile p"""
    tx = rp.profile.load("dvbs2/ts").tx()
    bits = np.unpackbits(np.frombuffer(b"".join(ts), np.uint8)).astype(bool)
    o = tx(rp.Data(bits, Fs=1e6, sps=1, flush=True), fields={"modcod": 4, "pilots": 1})
    return receive_ts(p, o, block)

# ---- PSI/SI sections (ISO/IEC 13818-1 2.4.4, EN 300 468 5.2, 6.2)

def crc32(body):
    return rp.link.load("crc/crc32_mpeg2").encoder.compute(body).to_bytes(4, "big")

def long_section(table_id, ext, body, version=1, number=0, last=0):
    """A section with section syntax: header, the table id extension and numbers, the body, the CRC-32"""
    n = len(body) + 9
    head = bytes([table_id, 0xB0 | n >> 8, n & 255]) + struct.pack(">H", ext) + bytes([0xC1 | version << 1, number, last])
    return head + body + crc32(head + body)

def descriptor(tag, body):
    return bytes([tag, len(body)]) + body

def loop(descriptors):
    """A descriptor loop with its 12-bit length (4 reserved bits set)"""
    d = b"".join(descriptors)
    return struct.pack(">H", 0xF000 | len(d)) + d

def bcd(v, digits):
    return int(str(v).zfill(digits), 16)

def mjd(date):
    return (datetime.date.fromisoformat(date) - datetime.date(1858, 11, 17)).days

def utc(date, time):
    """MJD then hh:mm:ss in BCD: 40 bits"""
    h, m, s = (int(x) for x in time.split(":"))
    return struct.pack(">H", mjd(date)) + bytes([bcd(h, 2), bcd(m, 2), bcd(s, 2)])

def network_name(name):
    return descriptor(0x40, name.encode("latin-1"))

def satellite_delivery(freq_hz, orbital_tenths, symbol_rate_bd, east=True, s2=True, fec=3):
    """Frequency in 10 kHz BCD digits, orbital position in tenths of a degree, symbol rate in 100 Bd digits; horizontal,
    roll-off 0.35, QPSK"""
    sr = bcd(symbol_rate_bd//100, 7) << 4 | fec
    return descriptor(0x43, struct.pack(">IH", bcd(freq_hz//10000, 8), bcd(orbital_tenths, 4))
                      + bytes([east << 7 | 0 << 5 | 0 << 3 | s2 << 2 | 1]) + struct.pack(">I", sr))

def service(service_type, provider, name):
    p, n = provider, name
    return descriptor(0x48, bytes([service_type, len(p)]) + p + bytes([len(n)]) + n)

def short_event(lang, name, text):
    return descriptor(0x4D, lang.encode() + bytes([len(name)]) + name.encode("latin-1") + bytes([len(text)])
                      + text.encode("latin-1"))

def local_time_offset(country, offset_min, change_date, change_time, next_min):
    h, m = divmod(offset_min, 60)
    nh, nm = divmod(next_min, 60)
    return descriptor(0x58, country.encode() + bytes([0x02]) + bytes([bcd(h, 2), bcd(m, 2)])
                      + utc(change_date, change_time) + bytes([bcd(nh, 2), bcd(nm, 2)]))

def pat(tsid, programs):
    body = b"".join(struct.pack(">HH", n, 0xE000 | pid) for n, pid in programs)
    return long_section(0x00, tsid, body)

def pmt(program, pcr_pid, streams):
    body = struct.pack(">H", 0xE000 | pcr_pid) + loop([])
    for stype, pid in streams:
        body += bytes([stype]) + struct.pack(">H", 0xE000 | pid) + loop([descriptor(0x52, bytes([pid & 255]))])
    return long_section(0x02, program, body)

def nit(network_id, name, streams):
    """streams: [(tsid, onid, [descriptors])]"""
    ts = b"".join(struct.pack(">HH", t, o) + loop(d) for t, o, d in streams)
    return long_section(0x40, network_id, loop([network_name(name)]) + struct.pack(">H", 0xF000 | len(ts)) + ts)

def sdt(tsid, onid, services):
    """services: [(service_id, running_status, [descriptors])]"""
    body = struct.pack(">H", onid) + b"\xff"
    for sid, running, descs in services:
        d = b"".join(descs)
        body += struct.pack(">HBH", sid, 0xFC | 0b01, running << 13 | 0x0000 | len(d)) + d
    return long_section(0x42, tsid, body)

def eit_pf(sid, tsid, onid, events):
    """events: [(event_id, date, start, duration, [descriptors])]"""
    body = struct.pack(">HHBB", tsid, onid, 1, 0x4E)
    for eid, date, start, dur, descs in events:
        h, m, s = (int(x) for x in dur.split(":"))
        d = b"".join(descs)
        body += struct.pack(">H", eid) + utc(date, start) + bytes([bcd(h, 2), bcd(m, 2), bcd(s, 2)])
        body += struct.pack(">H", 4 << 13 | len(d)) + d
    return long_section(0x4E, sid, body)

def tdt(date, time):
    """The TDT: no section syntax, no CRC"""
    return bytes([0x70, 0x70, 5]) + utc(date, time)

def tot(date, time, descriptors):
    """The TOT: no section syntax, and a CRC-32 all the same"""
    body = utc(date, time) + loop(descriptors)
    head = bytes([0x73, 0x70 | (len(body) + 4) >> 8, (len(body) + 4) & 255])
    return head + body + crc32(head + body)
