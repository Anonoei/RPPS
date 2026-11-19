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
    if v >> n:
        raise ConfigError(f"word {v:#x} is wider than {n} bits")
    idx = np.arange(n) if order == "lsb" else np.arange(n - 1, -1, -1)
    return ((v >> idx) & 1).astype(bool)

class Word:
    """A known bit word. `bits`: the bits as sent (bool array); `value`, `n`, `order` as defined"""
    def __init__(self, name, word, bits=None, order="msb"):
        if order not in ("msb", "lsb"):
            raise ConfigError(f"word {name}: order is msb or lsb, not {order!r}")
        self.name, self.order = name, order
        self.bits = word_bits(word, bits, order)
        self.n = len(self.bits)
        w = 1 << (np.arange(self.n) if order == "lsb" else np.arange(self.n - 1, -1, -1))
        self.value = int(self.bits.astype(np.int64) @ w)

    def __len__(self):
        return self.n

    def __repr__(self):
        return f"<Word {self.name}: {self.value:#x}, {self.n} bits {self.order} first>"

    @staticmethod
    def load(name, obj):
        if "word" not in obj:
            raise ConfigError(f"word {name}: needs 'word' (and 'bits' when it is a number)")
        return Word(name, obj["word"], obj.get("bits"), obj.get("order", "msb"))
