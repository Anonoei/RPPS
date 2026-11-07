"""Entry point for `python -m rpps` (also works from a source checkout without installing)"""
import sys

# NOTE: nothing but `sys` (builtin) and `os` (already loaded at startup) may be imported before
# the path fix below. `python src/rpps` puts this package dir on sys.path, where rpps/math,
# rpps/logger.py, etc. would shadow the stdlib and third-party modules.
import os


def _bootstrap():
    """Make `import rpps` work when run from a source tree (`python src/rpps`, `python -m src.rpps`)"""
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != here]
    src = os.path.dirname(here)
    if src not in sys.path:
        sys.path.insert(0, src)


if __name__ == "__main__":
    _bootstrap()
    from rpps.cli import main
    sys.exit(main())
