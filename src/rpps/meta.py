"""Pipeline metadata"""
import pathlib
import json

from . import __version__ as version

from .base._meta import _Meta, Construct
from .helpers import Formats


class Meta:
    freq = _Meta()
    inter = _Meta()
    mod = _Meta()
    coding = _Meta()

    def __str__(self):
        return ".".join([
            self.freq.__str__(),
            self.mod.__str__(),
            self.coding.__str__(),
        ])

    def short(self) -> str:
        lst = []
        lst.append(self.mod.short())
        lst.append(self.fmt)
        return ".".join(lst)

    def __repr__(self) -> str:
        return f"<Meta: {str(self)}>"

    def to_name(self):
        return f"{self.short()}.rpps"

    def json(self):
        return {
            "RPPS": version,
            "Freq": self.freq.fields,
            "Mod": self.mod.fields,
            "Coding": self.coding.fields,
        }

    def to_json(self):
        return json.dumps(
            self.json(),
            indent=4,
        )

    @staticmethod
    def from_json(d: dict):
        meta = Meta()
        # print(f"Creating Meta from {json.dumps(d, indent=4)}")
        meta.fmt = d["Format"]
        meta.freq = Construct("Freq", d["Freq"])
        meta.mod = Construct("Mod", d["Mod"])
        meta.coding = Construct("Coding", d["Coding"])
        return meta

    @staticmethod
    def read_json(path: pathlib.Path):
        return Meta.from_json(json.loads(path.read_text()))

    @staticmethod
    def from_name(name: str, meta = None):
        if meta is None:
            meta = Meta()
        fields = name.split(".")
        meta.mod = fields[0].upper()
        return meta
