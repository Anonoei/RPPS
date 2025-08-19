from ..modulation import Modulation

class ASK(Modulation):
    """Amplitude-shift keying parent"""
    def encode(self, data):
        raise NotImplementedError()
    def decode(self, data):
        raise NotImplementedError()
    @staticmethod
    def load(name: str, obj: dict):
        raise NotImplementedError()
