"""DVB service information out of a transport stream: the network, its services and the time

    python examples/dvb_si.py

The PSI/SI tables a DVB receiver reads to find its way round a multiplex (EN 300 468, ISO/IEC 13818-1) are sections
on known PIDs: the PAT on 0x00, the NIT on 0x10, the SDT on 0x11, the TDT and TOT on 0x14, the PMTs where the PAT says.
header/link/dvb_si reads every one of them (and builds them, transmitting): the section header, the table by its id,
its descriptors by tag, the CRC-32 checked where there is one. Here a NIT, an SDT, a TDT and a TOT are built from
values with that definition, carried in transport packets over DVB-S2 (dvbs2/ts), and received with dvbs2/si: a
message per section, its table in `table_id.name`, dates, times, frequencies and symbol rates decoded. DVB-S is the
same with dvbs/si.
"""
import numpy as np

import rpps as rp

def section(si, fields, payload):
    """A section's bytes: the table's payload built with its header's fields in scope, then header, payload and CRC-32,
    the lengths worked out"""
    body = si.payload_schema.build(payload, fields)
    si.control(**fields)
    return np.packbits(np.asarray((rp.Data(body, flush=True) * si).data, bool)).tobytes()

def packets(sec, pid):
    """One section in transport packets of a PID: a pointer field in the first, 0xFF filling the last"""
    data, out = b"\x00" + sec, []
    for cc, i in enumerate(range(0, len(data), 184)):
        head = bytes([0x47, (i == 0) << 6 | pid >> 8, pid & 255, 0x10 | cc % 16])
        out.append(head + data[i:i + 184].ljust(184, b"\xff"))
    return out

def descriptor(tag, **body):
    return {"tag": tag, "body": body}

def main():
    si = rp.link.load("header/dvb_si")
    syntax = {"ssi": 1, "version_number": 3}
    nit = section(si, {"table_id": 0x40, "table_id_extension": 0x0055, **syntax}, {"nit": {
        "descriptors": [descriptor(0x40, network_name="RPPS sat")],
        "transport_streams": [{"transport_stream_id": 1, "original_network_id": 0x0055, "descriptors": [
            descriptor(0x43, frequency=11778000000, orbital_position=192, west_east=1, modulation_system=1,
                       modulation_type=2, symbol_rate=27500000, **{"fec_inner.name": "3/4"})]}]}})
    sdt = section(si, {"table_id": 0x42, "table_id_extension": 1, **syntax}, {"sdt": {
        "original_network_id": 0x0055,
        "services": [{"service_id": 100, "eit_present_following": 1, "running_status": 4, "descriptors": [
            descriptor(0x48, service_type=25, provider_name="RPPS", service_name="Café HD")]}]}})
    tdt = section(si, {"table_id": 0x70, "ssi": 0}, {"tdt": {"utc_date": "2026-11-26", "utc_time": "12:34:56"}})
    tot = section(si, {"table_id": 0x73, "ssi": 0}, {"tot": {
        "utc_date": "2026-11-26", "utc_time": "12:34:56",
        "descriptors": [descriptor(0x58, offsets=[{"country_code": "DEU", "local_time_offset": "01:00",
                                                    "time_of_change_date": "2027-03-28", "time_of_change_time": "01:00:00",
                                                    "next_time_offset": "02:00"}])]}})
    ts = packets(nit, 0x10) + packets(sdt, 0x11) + packets(tdt, 0x14) + packets(tot, 0x14)
    ts += [bytes([0x47, 0x1F, 0xFF, 0x10]) + b"\xff"*184]*2 # (null packets: the last user packet's check needs one after)

    # Over DVB-S2 (TS mode), and back
    o = rp.profile.load("dvbs2/ts").tx()(rp.Data(np.unpackbits(np.frombuffer(b"".join(ts), np.uint8)).astype(bool),
                                                 Fs=1e6, sps=1, flush=True), fields={"modcod": 4, "pilots": 1})
    rx = rp.profile.load("dvbs2/si", sync={"timing": {"method": "known"}}).rx()
    rx(rp.Data(np.concatenate((o.data, np.zeros(200, np.complex64))), Fs=o.Fs, sps=4, delay=o.delay, flush=True))
    tables = {m.fields["table_id.name"]: m for m in rx.take_messages() if m.kind == "si"}

    assert set(tables) == {"NIT", "SDT", "TDT", "TOT"} and all(m.crc_ok is not False for m in tables.values())
    nit = tables["NIT"].fields["payload"]["nit"]
    name = nit["descriptors"][0]["body"]["network_name"]
    sat = nit["transport_streams"][0]["descriptors"][0]["body"]
    print(f"network {name}: {sat['frequency']/1e9:.3f} GHz, {sat['orbital_position']/10}° "
          f"{sat['west_east.name']}, {sat['modulation_system.name']} {sat['modulation_type.name']} "
          f"{sat['symbol_rate']/1e6} MBd FEC {sat['fec_inner.name']}")
    assert (name, sat["frequency"], sat["symbol_rate"]) == ("RPPS sat", 11778000000, 27500000)
    svc = tables["SDT"].fields["payload"]["sdt"]["services"][0]
    body = svc["descriptors"][0]["body"]
    print(f"service {svc['service_id']}: {body['service_name']} ({body['service_type.name']}, "
          f"{svc['running_status.name']}) from {body['provider_name']}")
    assert body["service_name"] == "Café HD"
    tdt = tables["TDT"].fields["payload"]["tdt"]
    off = tables["TOT"].fields["payload"]["tot"]["descriptors"][0]["body"]["offsets"][0]
    print(f"UTC {tdt['utc_date']} {tdt['utc_time']} (no CRC: crc_ok {tables['TDT'].crc_ok}); "
          f"{off['country_code']} +{off['local_time_offset']}, +{off['next_time_offset']} from {off['time_of_change_date']}")
    assert (tdt["utc_date"], tdt["utc_time"], off["local_time_offset"]) == ("2026-11-26", "12:34:56", "01:00")

if __name__ == "__main__":
    main()
