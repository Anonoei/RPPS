"""Wrapper for processing files"""

from .utils import Formats, Format


class file:
    fmt = Format

    @staticmethod
    def get_format():
        return file.fmt

    @staticmethod
    def read(fmt, path: str, count: int, offset: int = 0, skip=1):
        file.fmt = Formats[fmt]()
        for sym in file.fmt.read(count, path, offset, skip):
            yield sym

    @staticmethod
    def read_list(fmt, path: str, count: int, offset: int = 0, skip=1):
        return next(file.read(fmt, path, count, offset, skip))
