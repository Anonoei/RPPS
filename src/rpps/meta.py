import pathlib

class Meta:
    mod = ""
    fmt = ""
    fec = ""

    def serialize(self, data):
        path = pathlib.Path(f"data/{self.to_name()}")
        data.tofile(path) # Save to file
        return path

    def to_name(self):
        fields = [
            self.mod.lower(),
            self.fmt,
            "rpps",
            "iq",
        ]
        return ".".join(fields)

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
        meta = Meta.from_name(path.name, meta)
        symbols = np.fromfile(path, getattr(np, meta.fmt))
        return meta, symbols
