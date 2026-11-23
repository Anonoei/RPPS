"""Link transform: data bits <-> packets: framing (HDLC, byte stuffing) and integrity checks (CRC, composite, checksum), defined in
config/crc, config/frame and config/header/link. Decoding refuses channel bits: decode them first (or `rp.code.Uncoded()` for no FEC)
"""
from .crc import crc
from .hdlc import hdlc
from . import check
from .checksum import checksum
from .bytestuff import bytestuff
from .encap import Encap, CRC, Frame
from .header import Header
from .serial import ls, load, root
