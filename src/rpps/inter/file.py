"""Process data to/from files"""
from .interface import Interface

class File(Interface):
    """File interface"""
    def __init__(self, meta, pipeline, baud, r_file: str, w_file: str):
        super().__init__(meta, pipeline, baud)
        self._r = r_file
        self._w = w_file

    def _read(self) -> bytes | None:
        try:
            with open(self._r, "rb+") as f:
                data = f.read()
                input(data)
                f.truncate(0) # clear file contents
            return data
        except FileNotFoundError:
            return None

    def _write(self, data):
        with open(self._w, mode="ab") as f:
            f.write(data)
