"""Link transform: data bits <-> packets (ARCHITECTURE.md): framing and integrity checks (CRC, HDLC), defined in
config/encap/<kind>/<name>.json. Decoding refuses channel bits: decode them first (or `rp.code.Uncoded()` for no FEC)
"""
from .crc import crc
from .hdlc import hdlc
from .encap import Encap, CRC, Frame
from .header import Header
from .serial import ls, load, root
