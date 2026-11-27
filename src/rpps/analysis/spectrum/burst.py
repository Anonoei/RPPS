"""Moved to rpps.analysis.burst. `demodulate` (stable API) stays here for one release with a DeprecationWarning"""

def demodulate(*args, **kwargs):
    """rpps.analysis.burst.demodulate, under its old name: it goes in the next release"""
    import warnings
    from ..burst import demodulate as _demodulate
    warnings.warn("rp.analysis.spectrum.burst.demodulate moved to rp.analysis.burst.demodulate; the old name goes in "
                  "the next release", DeprecationWarning, stacklevel=2)
    return _demodulate(*args, **kwargs)
