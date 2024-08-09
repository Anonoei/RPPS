import time
from typing import overload

class progress:
    _name = ""
    _start = 0.0

    @staticmethod
    def _make_bar(name: str, percent: float, message: str="", chars=30, fill="X", blank=".") -> str:
        if not name == progress._name:
            progress._name = name
            progress._start = time.perf_counter()
        fill_chars = int(chars * percent)
        blank_chars = chars - fill_chars
        return f" |{fill * fill_chars}{blank * blank_chars}| "\
            f"{time.perf_counter() - progress._start:.2f}s "\
            f"{percent*100:.1f}%"


    @staticmethod
    def bar_percent(name: str, percent: float, message: str="", chars=30, fill="X", blank="."):
        return progress._make_bar(name, percent, message, chars, fill, blank) +\
            ": " \
            f"{message}"

    @staticmethod
    def bar(name: str, cur: int, tot: int, message: str="", chars=30, fill="X", blank="."):
        return progress._make_bar(name, (cur / tot), message, chars, fill, blank) +\
            f" {cur:.1f}/{tot:.1f}: "\
            f"{message}"
