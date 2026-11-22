"""Exceptions"""

class RPPSError(Exception):
    """Base of the RPPS exceptions"""

class StreamError(RPPSError):
    """Data does not fit the stage or stream it was given to (wrong type, rate or state)"""

class ConfigError(RPPSError, LookupError):
    """A definition (modulation, code, scrambler...) does not exist or is malformed"""

class NoLock(RPPSError):
    """No signal to lock to (`rpps.sync.Lock`): reason says why, line is the symbol-rate line's strength when one was measured"""
    def __init__(self, reason, line=None):
        super().__init__(reason)
        self.reason, self.line = reason, line
