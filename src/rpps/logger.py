"""Logging on the standard library

RPPS is silent by default (a NullHandler on the "rpps" logger). To see its log:

    rp.logger.enable("DEBUG")                  # console
    rp.logger.enable("TRACE", file="rpps.log") # console and file
"""
import logging

TRACE = 5
logging.addLevelName(TRACE, "TRACE")

_ROOT = "rpps"
logging.getLogger(_ROOT).addHandler(logging.NullHandler())
_FORMAT = logging.Formatter("%(asctime)s <%(name)s> %(levelname)s: %(message)s", datefmt="%Y%m%d_%H%M%S")

class Logger:
    """A named logger. Children are named parent.child"""
    __slots__ = ("_log",)
    def __init__(self, name=_ROOT, level=None):
        self._log = logging.getLogger(name if name.startswith(_ROOT) else f"{_ROOT}.{name}")
        if level is not None and self._log.level == logging.NOTSET:
            self._log.setLevel(level)

    @property
    def name(self):
        return self._log.name

    def Child(self, name, level=None):
        return Logger(f"{self._log.name}.{name}", level)

    def trace(self, msg, *args):
        if self._log.isEnabledFor(TRACE):
            self._log.log(TRACE, msg, *args)
    def debug(self, msg, *args):
        self._log.debug(msg, *args)
    def info(self, msg, *args):
        self._log.info(msg, *args)
    def warn(self, msg, *args):
        self._log.warning(msg, *args)
    warning = warn
    def error(self, msg, *args):
        self._log.error(msg, *args)
    def exception(self, msg, *args):
        self._log.exception(msg, *args)

def enable(level="INFO", file=None, console=True):
    """Send the RPPS log to the console and/or a file. level: a name (TRACE..ERROR) or number"""
    root = logging.getLogger(_ROOT)
    if isinstance(level, str):
        level = {"TRACE": TRACE}.get(level.upper(), None) or getattr(logging, level.upper())
    root.setLevel(level)
    for h in list(root.handlers):
        if getattr(h, "_rpps", False):
            root.removeHandler(h)
    handlers = []
    if console:
        handlers.append(logging.StreamHandler())
    if file:
        handlers.append(logging.FileHandler(file))
    for h in handlers:
        h.setFormatter(_FORMAT)
        h._rpps = True # type: ignore
        root.addHandler(h)

def disable():
    """Back to silent"""
    enable_root = logging.getLogger(_ROOT)
    for h in list(enable_root.handlers):
        if getattr(h, "_rpps", False):
            enable_root.removeHandler(h)
    enable_root.setLevel(logging.NOTSET)
