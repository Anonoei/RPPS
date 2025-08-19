"""
Example showing encoding and decoding data
"""
import rpps as rp

mod = rp.mod.load("QPSK")
mod.set_mapping(mod.get_maps()[0])
ecc = rp.coding.load("blk", "hamming.7_4")
scr = rp.scram.load("fdt", "v35")

print(f"Mod: {mod}")
print(f"ECC: {ecc}")
print(f"SCR: {scr}")

enc_msg = rp.Data(b"Hello World!")

syms = enc_msg(scr+ecc+mod)*scr*ecc*mod # Scrambled, encoding, and modulate
print(f" Symbols: {syms}")
dec_msg = syms(mod-ecc-scr)/mod/ecc/scr # Demodulate, decode, and descramble

enc_msg = enc_msg.as_bytes()
dec_msg = dec_msg.as_bytes()
print(f"{enc_msg.hex == dec_msg.hex}") # Check decoded data is what you encoded
