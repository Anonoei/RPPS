"""Encapsulation parent classes"""
from ..logger import Logger
from ..errors import ConfigError

from .. import _config
from ..coding.coding import Coding
from .crc import crc as _crc_code
from .hdlc import hdlc as _hdlc_code


class Encap(Coding):
    """Parent of framing / integrity-check definitions"""
    kind = "encap"

    @staticmethod
    def load(name: str, obj: dict):
        """Load from json: the `type` picks CRC or HDLC"""
        t = obj.get("type")
        if t == "crc":
            return CRC.load(name, obj)
        if t == "hdlc":
            return Frame.load(name, obj)
        raise ConfigError(f"Unknown encapsulation type {t!r} in {name}: have crc, hdlc")


def _int(obj, key, default=0):
    v = obj.get(key, default)
    return int(v, 0) if isinstance(v, str) else v


class CRC(Encap):
    """CRC defined by config: width, poly, init, refin, refout, xorout (hex strings or ints), check value"""

    @staticmethod
    def load(name, obj):
        log = Logger().Child("Coding", _config.LOG_CODING).Child(name)
        i_code = _crc_code(log, obj["width"], _int(obj, "poly"), _int(obj, "init"), obj.get("refin", False),
                          obj.get("refout", False), _int(obj, "xorout"), _int(obj, "check") if "check" in obj else None)
        impl = type(name, (CRC,), dict())
        impl.name = name
        return impl(i_code, i_code)


class Frame(Encap):
    """Framing (HDLC) defined by config; the FCS is another config, `fcs`, from config/encap/crc"""

    @staticmethod
    def load(name, obj):
        from . import serial
        log = Logger().Child("Coding", _config.LOG_CODING).Child(name)
        fcs = serial.load("crc", obj["fcs"])._enc
        i_code = _hdlc_code(log, fcs, _int(obj, "flag", 0x7E), obj.get("stuff", 5), obj.get("nrzi", False), obj.get("min_bytes", 4))
        impl = type(name, (Frame,), dict())
        impl.name = name
        return impl(i_code, i_code)
