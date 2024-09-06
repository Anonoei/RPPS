from ..base._meta import _Meta

from collections import OrderedDict

class ModMeta(_Meta):
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
    meta = ModMeta()
    meta.from_json(j)
    return meta
