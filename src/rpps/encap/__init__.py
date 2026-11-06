"""Encapsulation: framing and integrity checks (CRC, HDLC) defined in config/encap/<kind>/<name>.json"""
from .crc import crc
from .hdlc import hdlc
from .encap import Encap, CRC, Frame
from .serial import ls, load, root
