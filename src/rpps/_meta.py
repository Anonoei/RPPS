import json

class _Meta:
    fields = {}

    def __str__(self):
        return "_".join([str(item) for item in self.fields.values()])

    def __repr__(self):
        return f"<_Meta: {_Meta.__str__(self)}>"

    def json(self):
        return self.fields

    def to_json(self):
        return json.dumps(self.fields, indent=4)

    def from_json(self, j: dict):
        for k, v in j.items():
            self.fields[k]  # Check if passed dict matches self
            self.fields[k] = v

def Construct(name, j: dict):
    if name == "Mod":
        from .mod.meta import Construct
    elif name == "Coding":
        from .coding.meta import Construct
    return Construct(j)
