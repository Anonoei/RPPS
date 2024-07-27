from .._meta import _Meta

from collections import OrderedDict

class CodingMeta(_Meta):
    __slots__ = ("fields")

    def __init__(self):
        self.fields = OrderedDict((
            ("Type", None),
            ("Name", None),
            ("RateNum", None),
            ("RateDen", None),
        ))

    def short(self) -> str:
        return f"{self.fields['Name']}"


class BlockCodingMeta(CodingMeta):
    __slots__ = ("fields")
    def __init__(self):
        self.fields = OrderedDict(
            (
                ("Type", None),
                ("Name", None),
                ("RateNum", None),
                ("RateDen", None),
                ("Length", None),
            )
        )


class ConvolutionalCodingMeta(CodingMeta):
    __slots__ = ("fields")

def Construct(j: dict):
    if j.get("Type", None) is None:
        return CodingMeta()
    if j["Type"] == "BLK":
        meta = BlockCodingMeta()
    elif j["Type"] == "CNV":
        meta = ConvolutionalCodingMeta()
    meta.from_json(j)
    return meta
