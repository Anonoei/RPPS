from .._meta import _Meta

from collections import OrderedDict

class ModMeta(_Meta):
    __slots__ = "fields"

    def __init__(self):
        self.fields = OrderedDict((
            ("Name", None),
            ("Type", None),
            ("Map", None),
        ))

    def __str__(self):
        return f"{self.fields['Name']}_{self.fields['Type']}_{self.fields['Map']}"

    def short(self) -> str:
        return f"{self.fields['Name']}{self.fields['Type']}"


def Construct(j: dict):
    meta = ModMeta()
    meta.from_json(j)
    return meta
