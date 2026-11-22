"""Bits to symbols and back: the README's first example

    python examples/quickstart.py
"""
import rpps as rp

mod = rp.symbol.load("psk/qpsk")    # first bit map; load("psk/qpsk", "R090 inverted") for a rotated / spectrally inverted one
ecc = rp.code.Hamming(7, 4)         # or rp.code.load("block/hamming/7_4")
scr = rp.code.scram.load("v35")

msg = rp.Data(b"Hello World!")
syms = msg * scr * ecc * mod        # scramble, encode, modulate
out = syms / mod / ecc / scr        # demodulate, decode, descramble
assert out.as_bytes().hex == msg.as_bytes().hex

# Pipes multiplied together make a reusable link, decoded in the opposite order
link = scr * ecc * mod
out = (msg * link) / link
assert out.as_bytes().hex == msg.as_bytes().hex
print("decoded:", out.as_bytes().hex)
