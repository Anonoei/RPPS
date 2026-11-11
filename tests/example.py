"""
Example showing encoding and decoding data
"""
import rpps as rp

mod = rp.symbol.load("QPSK")   # starts on its first bit map; rp.symbol.load("QPSK", "R090") picks another
ecc = rp.code.load("block", "hamming/7_4")
scr = rp.code.scram.load("fdt", "v35")

print(f"Mod: {mod}")
print(f"ECC: {ecc}")
print(f"SCR: {scr}")

enc_msg = rp.Data(b"Hello World!")

syms = enc_msg * scr * ecc * mod # Scrambled, encoding, and modulate
print(f" Symbols: {syms}")
dec_msg = syms / mod / ecc / scr # Demodulate, decode, and descramble

enc_msg = enc_msg.as_bytes()
dec_msg = dec_msg.as_bytes()
print(f"{enc_msg.hex == dec_msg.hex}") # Check decoded data is what you encoded
