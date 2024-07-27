import pathlib
import json

from . import __version__ as version

from ._meta import _Meta, Construct


class Meta:
    mod = _Meta()
    coding = _Meta()
    fmt = "complex64"

    def __str__(self):
        return ".".join([
            self.mod.__str__(),
            self.coding.__str__(),
            self.fmt,
        ])

    def short(self) -> str:
        lst = []
        lst.append(self.mod.short())
        lst.append(self.fmt)
        return ".".join(lst)

    def __repr__(self) -> str:
        return f"<Meta: {str(self)}>"

    def serialize(self, data):
        path = pathlib.Path(f"data/{self.to_name()}")
        meta_path = pathlib.Path(f"data/{self.to_name()}.meta")
        data.tofile(path) # Save IQ to file
        meta_path.write_text(self.to_json(), encoding="UTF-8")
        return path

    def to_name(self):
        return f"{self.short()}.rpps"

    def json(self):
        return {
            "RPPS": version,
            "Format": self.fmt,
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
        meta.fmt = fields[1]
        return meta

    @staticmethod
    def from_file(path, meta = None):
        import numpy as np
        if isinstance(path, str):
            path = pathlib.Path(path)
        meta_json = pathlib.Path(f"{str(path)}.meta")
        if meta is None:
            if meta_json.exists():
                meta = Meta.read_json(meta_json)
            else:
                meta = Meta.from_name(path.name, meta)
        # print(f"Meta is {str(meta)}")
        symbols = np.fromfile(path, getattr(np, meta.fmt))
        return meta, symbols
