"""Exceptions"""

class RPPSError(Exception):
    """Base of the RPPS exceptions"""

class DecodeError(RPPSError, ValueError):
    """A block could not be decoded (more errors than the code corrects)"""

class StreamError(RPPSError):
    """Data does not fit the stage or stream it was given to (wrong type, rate or state)"""

class ConfigError(RPPSError, LookupError):
    """A definition (modulation, code, scrambler...) does not exist or is malformed"""
