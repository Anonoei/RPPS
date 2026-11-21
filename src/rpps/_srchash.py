"""Hash of the C++ kernel sources, so a dev checkout can tell when rpps._native was built from other sources

setup.py loads this file by path (it imports nothing from rpps) and compiles the hash into the extension as
`_native.SRC_HASH`; rpps._accel compares it with the sources in the checkout on import.
"""
import hashlib
from pathlib import Path

def source_hash(src_dir):
    """First 16 hex digits of the SHA-256 over the names and contents of the .cpp and .hpp files in src_dir"""
    h = hashlib.sha256()
    for p in sorted(Path(src_dir).iterdir()):
        if p.suffix in (".cpp", ".hpp"):
            h.update(p.name.encode())
            h.update(p.read_bytes().replace(b"\r\n", b"\n")) # The same hash for a CRLF checkout (Windows)
    return h.hexdigest()[:16]
