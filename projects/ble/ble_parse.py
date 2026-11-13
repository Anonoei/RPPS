"""Bluetooth LE advertising PDU parsing: addresses, AD structures, and what kind of device is advertising

Core Specification v6.0 Vol 6 Part B 2.3 (advertising PDUs) and the Core Specification Supplement (AD types). A legacy advertising PDU payload is
AdvA (6 bytes, least significant byte first) followed by AdvData, a list of AD structures [length][type][data]. A random address's two most significant bits
give its kind: 11 static random, 01 resolvable private (rotates about every 15 minutes, so unique-address counts overestimate the number of devices), 00 non-resolvable.
"""
import hashlib

PDU_TYPES = {0: "ADV_IND", 1: "ADV_DIRECT_IND", 2: "ADV_NONCONN_IND", 3: "SCAN_REQ", 4: "SCAN_RSP", 5: "CONNECT_IND", 6: "ADV_SCAN_IND", 7: "ADV_EXT_IND"}
# company identifiers (Bluetooth SIG assigned numbers): only the common ones; others print as hex
COMPANIES = {0x004C: "Apple", 0x0006: "Microsoft", 0x00E0: "Google", 0x0075: "Samsung", 0x0087: "Garmin", 0x0059: "Nordic Semiconductor", 0x02E5: "Espressif",
             0x0171: "Amazon", 0x0157: "Anhui Huami (Amazfit)", 0x0499: "Ruuvi", 0x004F: "APT/Qualcomm", 0x000F: "Broadcom", 0x00D2: "Dialog", 0x0131: "Cypress",
             0x0310: "Sonos?", 0x038F: "Xiaomi", 0x0A12: "CSR?", 0x01D7: "Qualcomm", 0x0001: "Nokia", 0x0002: "Intel", 0x0003: "IBM", 0x0008: "Motorola",
             0x001D: "Qualcomm", 0x0046: "MediaTek", 0x00C4: "LG", 0x0078: "Nike", 0x0102: "Dexcom?", 0x011B: "Fitbit?", 0x0822: "Tile?"}
APPLE_TYPES = {0x02: "iBeacon", 0x05: "AirDrop", 0x07: "proximity pairing (AirPods etc.)", 0x08: "Hey Siri", 0x09: "AirPlay target", 0x0A: "AirPlay source", 0x0B: "magic switch",
               0x0C: "handoff", 0x0D: "tethering target", 0x0E: "tethering source", 0x0F: "nearby action", 0x10: "nearby info", 0x12: "Find My (offline finding)", 0x16: "nearby action (new)"}
SERVICES16 = {0xFEAA: "Eddystone", 0xFD6F: "Exposure Notification", 0xFEED: "Tile", 0xFD84: "Tile", 0xFE2C: "Google Fast Pair", 0xFEF3: "Google", 0xFE9F: "Google",
              0xFE95: "Xiaomi", 0xFD5A: "Samsung SmartTag", 0xFD69: "Samsung", 0xFEBE: "Bose", 0xFE59: "Nordic DFU", 0x180F: "Battery", 0x180A: "Device Information",
              0x181A: "Environmental sensing", 0x1812: "HID", 0x180D: "Heart rate", 0xFE78: "Hunter Fan", 0xFCF1: "Google", 0xFEE0: "Huami", 0xFEE7: "Tencent", 0xFFF0: "vendor"}
AD_NAMES = {0x01: "flags", 0x02: "uuid16 (partial)", 0x03: "uuid16", 0x06: "uuid128 (partial)", 0x07: "uuid128", 0x08: "short name", 0x09: "name", 0x0A: "tx power",
            0x16: "service data", 0x19: "appearance", 0xFF: "manufacturer data"}

def addr_str(b):
    return ":".join(f"{v:02x}" for v in reversed(b))

def addr_kind(b, txadd):
    if not txadd:
        return "public"
    top = b[5] >> 6
    return {3: "random static", 1: "resolvable private", 0: "non-resolvable private"}.get(top, "random (reserved)")

def short_id(b):
    return hashlib.sha1(bytes(b)).hexdigest()[:4]

def ad_structures(data):
    out = []; i = 0
    while i < len(data):
        n = data[i]
        if n == 0 or i + 1 + n > len(data):
            break
        out.append((data[i + 1], bytes(data[i + 2:i + 1 + n])))
        i += 1 + n
    return out

def parse(r):
    """r: dict from ble_adv.decode_burst with a valid CRC -> dict(kind, addr (bytes LSB first), addr_kind, ad, company, labels, name)"""
    t = r["type"]; p = r["payload"]
    d = dict(kind=PDU_TYPES.get(t, f"type {t}"), length=r["length"], labels=[], company=None, name=None, ad=[])
    if t in (0, 1, 2, 4, 6) and len(p) >= 6:
        d["addr"] = p[:6]; d["addr_kind"] = addr_kind(p[:6], r["txadd"])
        if t != 1:
            d["ad"] = ad_structures(p[6:])
    elif t == 3 and len(p) >= 12: # SCAN_REQ: scanner address, then the advertiser's
        d["addr"] = p[6:12]; d["addr_kind"] = addr_kind(p[6:12], r["rxadd"]); d["scanner"] = addr_kind(p[:6], r["txadd"])
    elif t == 5 and len(p) >= 12:
        d["addr"] = p[6:12]; d["addr_kind"] = addr_kind(p[6:12], r["rxadd"]); d["initiator"] = addr_kind(p[:6], r["txadd"])
    for typ, v in d["ad"]:
        if typ == 0xFF and len(v) >= 2:
            cid = v[0] | (v[1] << 8)
            d["company"] = cid
            d["labels"].append(COMPANIES.get(cid, f"company 0x{cid:04x}"))
            if cid == 0x004C and len(v) >= 3:
                d["labels"].append("Apple: " + APPLE_TYPES.get(v[2], f"type 0x{v[2]:02x}"))
        elif typ in (0x02, 0x03):
            for j in range(0, len(v) - 1, 2):
                u = v[j] | (v[j + 1] << 8)
                if u in SERVICES16: d["labels"].append(SERVICES16[u])
        elif typ == 0x16 and len(v) >= 2:
            u = v[0] | (v[1] << 8)
            d["labels"].append(SERVICES16.get(u, f"service data 0x{u:04x}"))
        elif typ in (0x08, 0x09):
            d["name"] = v.decode("utf8", "replace")
    if d["kind"] == "ADV_EXT_IND":
        d["labels"].append("extended advertising")
    return d
