"""Link transform: data bits <-> packets: framing (HDLC, byte stuffing) and integrity checks (CRC, composite, checksum), defined in
config/crc, config/frame and config/header/link. Decoding refuses channel bits: decode them first (or `rp.code.Uncoded()` for no FEC)

A link header decodes packets too: a header inside a frame's payload (`{"link": "frame"}` then `{"link": "header"}`, the
frame step's `frame` and `frame_bits` driving the header's `frame` and `length`) strips a frame's own header and reads it.

Packets that cross frames (`rpps.link.packets`, `{"link": "packets"}`): a stream of packets in fixed frames' data fields,
cut back out by a pointer (CCSDS first header pointer, DVB-S2 SYNCD) and each packet's length, per channel.

Captures (`rpps.link.capture`): a profile's messages as packets of a link type (config/linktype), for Wireshark and
interfaces (rpps.source.PcapSink, rpps.source.Tun), each rule with its inverse for transmitting.
"""
from .crc import crc
from .hdlc import hdlc
from . import check
from .checksum import checksum
from .bytestuff import bytestuff
from .encap import Encap, CRC, Frame
from .header import Header
from .serial import ls, load, root
from . import capture
from .capture import LinkType, linktype, Capture, Captures, Packet
