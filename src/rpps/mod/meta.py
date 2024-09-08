"""Modulation meta helpers"""
from ..base._meta import _Meta

from collections import OrderedDict

class ModMeta(_Meta):
    """Modulation meta class"""
    __slots__ = "fields"

    def __init__(self):
        self.fields = OrderedDict((
            ("Name", None),
            ("Type", None),
            ("Map", None),
        ))

    def short(self) -> str:
        return f"{self.fields['Name']}{self.fields['Type']}"


def Construct(j: dict):
    """Construct ModMeta from json"""
    meta = ModMeta()
    meta.from_json(j)
    return meta
