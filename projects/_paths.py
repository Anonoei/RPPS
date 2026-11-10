"""Make every program in projects/ importable from any other, whichever folder it runs from

A script in a sub-folder starts with the one line below (the folder holding this file must already be on sys.path to import it); after that
`import radio_io`, `import p25_sync`, `import atsc_probe` ... work wherever the module lives:

    import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); import _paths  # noqa: E402,F401

Every project folder is one level down, so the line is the same everywhere. Module names are unique across the folders.
"""
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))

def folders():
    """projects/ and every folder under it that holds Python modules"""
    out = [ROOT]
    for d, subs, files in os.walk(ROOT):
        subs[:] = sorted(s for s in subs if not s.startswith((".", "__")))
        if d != ROOT and any(f.endswith(".py") for f in files):
            out.append(d)
    return out

for _d in folders():
    if _d not in sys.path:
        sys.path.append(_d)
