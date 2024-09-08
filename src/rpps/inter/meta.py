"""Interface meta helpers"""
from ..base._meta import _Meta

from collections import OrderedDict


class InterMeta(_Meta):
    """Interface meta class"""
    __slots__ = "fields"

    def __init__(self):
        self.fields = OrderedDict(
            (
                ("Baud", None),
            )
        )

    def short(self) -> str:
        return f"{self.fields['Baud']}"


def Construct(j: dict):
    """Generate interface meta from json"""
    meta = InterMeta()
    meta.from_json(j)
    return meta
