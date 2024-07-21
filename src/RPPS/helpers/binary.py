# Encoding

def bytes_to_hex(data: bytes):
    return data.hex()

def hex_to_bits(data):
    bits = ""
    for i in range(0, len(data), 2):
        int_data = int(data[i:i+2], 16)
        bit_data = bin(int_data)[2:]
        padding = (len(bit_data) % 8)
        if not padding == 0:
            bit_data = "0" * (8 - padding) + bit_data
        bits += bit_data
        #input(f"0x{data[i:i+2]} = 0i{int_data}, 0b{bit_data}")
    return bits

def bytes_to_bits(data):
    return hex_to_bits(bytes_to_hex(data))


# Decoding

def bits_to_hex(bits):
    data_hex = ""
    for i in range(0, len(bits), 8):
        int_data = int(bits[i:i+8], 2)
        hex_data = hex(int_data)[2:]
        padding = (len(hex_data) % 2)
        if not padding == 0:
            hex_data = "0" * (2 - padding) + hex_data
        data_hex += hex_data
        #input(f"0b{bits[i:i+8]} = 0i{int_data}, 0x{hex_data}")
    return data_hex

def hex_to_bytes(data):
    return bytes.fromhex(data)

def bits_to_bytes(data):
    return hex_to_bytes(bits_to_hex(data))

class Encoding:
    pass
