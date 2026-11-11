from ..base import Mod, Digital, dtype
from ..base.rpps import Pipe
from ..base.types import Category, Rep
from ..base.data import expect

class Uncoded(Pipe):
    """No channel code: the channel bits are the data bits. The code step of a link that has no FEC (ADS-B, many
    packet radios), so that what the demodulator gives becomes what the link layer takes

        payload = syms / bpsk / rp.code.Uncoded() / crc

    Decoding keeps LLRs as LLRs (and hard bits as bits), only saying they are now data bits.
    """
    CATEGORY = Category.CODE

    def __str__(self):
        return "Uncoded"

    def pos_rate(self, decode):
        return 1.0

    def takes(self, encode):
        return {Rep.DATA, Rep.CHANNEL} if encode else {Rep.CHANNEL}

    def encode(self, data):
        return data.derive(data.bits(), Category.CODE, dtype.BITS, Digital, rep=Rep.CHANNEL)

    def decode(self, data):
        if isinstance(data, Mod) and data.is_soft:
            return data.derive(data.data, Category.CODE, dtype.LLR, Mod, rep=Rep.DATA)
        return data.derive(data.bits(), Category.CODE, dtype.BITS, Digital, rep=Rep.DATA)

    def __rmul__(self, data):
        expect(self, data, self.takes(True), True)
        return self.encode(data)

    def __rtruediv__(self, data):
        expect(self, data, self.takes(False), False)
        return self.decode(data)
