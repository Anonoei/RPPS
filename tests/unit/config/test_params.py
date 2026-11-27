"""Definition parameters and expressions"""
import pytest

from rpps import _config
from rpps.errors import ConfigError

# --- parameters -----------------------------------------------------------------------------------------------

def test_param_and_expr_substitution():
    obj = {"params": {"a": {"default": 3}, "b": None}, "x": {"param": "a"}, "y": [{"expr": "(a << 2) | (b >> 2)"}],
           "z": {"each": "i*a % 5", "n": {"param": "b"}}}
    out = _config.resolve(obj, {"b": 4})
    assert (out["x"], out["y"], out["z"]) == (3, [13], [0, 3, 1, 4])

def test_param_errors():
    obj = {"params": {"pci": None}, "seed": {"param": "pci"}}
    with pytest.raises(ConfigError, match="needed"):
        _config.resolve(obj, {})
    with pytest.raises(ConfigError, match="unknown parameter"):
        _config.resolve(obj, {"pcl": 3})
    assert _config.resolve(obj, {"pcl": 3, "pci": 1}, strict=False)["seed"] == 1

@pytest.mark.parametrize("expr", ["__import__('os')", "(1).real", "open('x')", "[x for x in range(3)]", "lambda: 1"])
def test_expressions_are_not_python(expr):
    with pytest.raises(ConfigError):
        _config.evaluate(expr, {})

def test_expression_operators():
    v = {"w": 0, "n": 5}
    assert _config.evaluate("w or 2**44 - 1", v) == 2**44 - 1
    assert _config.evaluate("[3, 4, 5][n % 3] if n > 2 else -1", v) == 5
    assert _config.evaluate("max(n, 7) // 2 + abs(-1) + (~n & 3)", v) == 6
