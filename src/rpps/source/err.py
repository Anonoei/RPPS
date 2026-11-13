from ..errors import RPPSError

class Overflow(RPPSError):
    """Read past the end of a source"""
