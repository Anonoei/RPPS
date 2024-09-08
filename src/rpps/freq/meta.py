"""Freq domain metadata helpers"""
from ..base._meta import _Meta

from collections import OrderedDict

class FreqMeta(_Meta):
    """Parent frequency metadata"""
    __slots__ = "fields"

    def __init__(self):
        self.fields = OrderedDict((
            ("SampleRate", None),
            ("CenterFreq", None),
            ("BlockSize", None),
            ("BlockTime", None),
        ))

    def short(self) -> str:
        return f"{self.fields['CenterFreq']}_{self.fields['SampleRate']}"


def Construct(j: dict):
    meta = FreqMeta()
    meta.from_json(j)
    return meta
