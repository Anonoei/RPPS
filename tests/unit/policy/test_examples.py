"""The worked examples in examples/ run and their own checks pass

Each one is a script that asserts what it shows; this runs it as `python examples/<name>.py` would. The examples
that need a capture, a radio or a display (e2e, e2e_mat, e2e_profile, live_profile, window) are not run here."""
import runpy

import pytest
from support.paths import ROOT

EXAMPLES = ROOT/"examples"

FAST = ["quickstart", "list_definitions", "operators", "soft_decisions", "channel_and_sync", "profile_loopback",
        "own_definitions", "profile_variants", "streaming_io", "integrating", "channel_shm", "zc_modes", "mftdma",
        "finding_signals", "wireshark_packets", "dvb_si"]
KERNELS = ["parallel_rx"] # seconds with the C++ kernels, minutes without
SLOW = ["unknown_signal"] # tens of seconds even with them

def run(name, monkeypatch):
    path = str(EXAMPLES/f"{name}.py")
    monkeypatch.setattr("sys.argv", [path]) # the script's own defaults, not pytest's arguments
    runpy.run_path(path, run_name="__main__")

@pytest.mark.parametrize("name", FAST)
def test_example(name, monkeypatch, config_layer): # config_layer: some examples add one
    run(name, monkeypatch)

@pytest.mark.kernels
@pytest.mark.parametrize("name", KERNELS)
def test_kernel_example(name, monkeypatch, config_layer): # config_layer: some examples add one
    run(name, monkeypatch)

@pytest.mark.slow
@pytest.mark.kernels
@pytest.mark.parametrize("name", SLOW)
def test_slow_example(name, monkeypatch, config_layer): # config_layer: some examples add one
    run(name, monkeypatch)

def test_every_example_is_run_or_listed():
    """A new example joins one of the lists above (or the ones that need hardware, here)"""
    needs_more = {"e2e", "e2e_mat", "e2e_profile", "live_profile", "window"}
    have = {p.stem for p in EXAMPLES.glob("*.py")}
    assert have == set(FAST) | set(KERNELS) | set(SLOW) | needs_more
