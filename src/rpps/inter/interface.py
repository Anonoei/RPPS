"""Defines interface bindings"""
import threading
import time

from .meta import InterMeta
#from .. import Stream

class Interface:
    """Parent interface class"""
    def __init__(self, meta, pipeline, baud=None):
        self.meta = meta
        self.pipeline = pipeline

        self._r_thread = threading.Thread(target=self.read)
        self._w_thread = threading.Thread(target=self.baud)
        self._r_buffer = []

        if meta.inter.get("Baud", None) is None:
            meta.inter = InterMeta()
            meta.inter["Baud"] = baud

        if meta.inter.fields.get("Baud", None) is None:
            raise Exception("No baud rate specified!")

    def start(self):
        """Start read/write threads"""
        self._r_thread.start()
        self._w_thread.start()

    def read(self):
        """Check and read new data"""
        while True:
            data = self._read()
            if data is None:
                continue
            input(f"Got data {data}!")
            self.pipeline.dec(data)
            self._r_buffer.insert(0, data)
            print(f"Buffer is {len(self._r_buffer)}: {self._r_buffer[0]}")

    def baud(self):
        """Write data at baud rate"""
        while True:
            time.sleep(self.meta.inter["Baud"]/1000)
            # self._write(self.pipeline.enc(Stream.from_bytes(0x7E.to_bytes())))

    def write(self, data):
        """Write data to interface"""
        self._write(self.pipeline.enc(data))

    def _read(self):
        return None

    def _write(self, data):
        return None
