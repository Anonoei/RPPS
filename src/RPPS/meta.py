class Meta:
    mod = ""
    fmt = ""
    fec = ""

    def serialize(self, data):
        data.tofile(f"data/{self.to_name()}") # Save to file

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
