"""scipy is a test oracle only: nothing under src/ may import it"""
import ast
import pathlib

import rpps as rp

def test_rpps_never_imports_scipy():
    """scipy is a test oracle only: nothing under src/ may import it"""
    for path in pathlib.Path(rp.__file__).parent.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""] if isinstance(node, ast.ImportFrom) else []
            assert not any(n.split(".")[0] == "scipy" for n in names), f"{path} imports scipy"
