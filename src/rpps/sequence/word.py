"""A fixed bit word (a sync marker, an access address, a start-of-frame delimiter), as a sequence definition

    {"base": "sequence", "type": "word", "word": "0x1ACFFC1D", "bits": 32, "order": "msb"}

`bits` is the word in the order it is sent: most significant bit first ("msb", the default) or least ("lsb", Bluetooth).
A delimiter search (rp.sync.Delimiter) and a marker (rp.code.Marker) name one, so the two agree.
"""
import numpy as np

from ..errors import ConfigError

def word_bits(word, bits=None, order="msb"):
    """The bits of a word in transmission order: from an int (or hex string) of `bits` bits, or from a list/array of bits as given"""
    if isinstance(word, (list, tuple, np.ndarray)):
        return np.asarray(word).astype(bool)
    v = int(word, 0) if isinstance(word, str) else int(word)
    if bits is None:
        raise ConfigError("a word given as a number needs its width, `bits`")
    n = int(bits)
    if v < 0 or v >> n:
        raise ConfigError(f"word {v:#x} is wider than {n} bits")
    msb = np.unpackbits(np.frombuffer(v.to_bytes((n + 7)//8, "big"), np.uint8))[-n:] if n else np.zeros(0, np.uint8) # (any width)
    return (msb[::-1] if order == "lsb" else msb).astype(bool)

class Word:
    """A known bit word. `bits`: the bits as sent (bool array); `value`, `n`, `order` as defined"""
    def __init__(self, name, word, bits=None, order="msb"):
        if order not in ("msb", "lsb"):
            raise ConfigError(f"word {name}: order is msb or lsb, not {order!r}")
        self.name, self.order = name, order
        self.bits = word_bits(word, bits, order)
        self.n = len(self.bits)
        msb = self.bits[::-1] if order == "lsb" else self.bits
        self.value = int("".join("1" if b else "0" for b in msb), 2) if self.n else 0 # (a Python int: any width)

    def __len__(self):
        return self.n

    def __repr__(self):
        return f"<Word {self.name}: {self.value:#x}, {self.n} bits {self.order} first>"

    @staticmethod
    def load(name, obj):
        if "word" not in obj:
            raise ConfigError(f"word {name}: needs 'word' (and 'bits' when it is a number)")
        return Word(name, obj["word"], obj.get("bits"), obj.get("order", "msb"))
