"""Spreading codes: Barker, Gold (GPS C/A), PN sets (802.15.4), from config/spread

    code = rp.spread.load("gps_ca", prn=1)    a Code: code.chips is (count, length) of +/-1

The operators that use them are elsewhere, by category: `rp.symbol.Spread` and `rp.symbol.Despread` (symbol transform),
`rp.sync.acquire_code` (code-phase and Doppler search). These codes become the `sequence` resource (ARCHITECTURE.md).
"""
from .code import Code
from .serial import load, init, ls
