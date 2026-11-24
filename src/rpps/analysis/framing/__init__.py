"""How the stream is cut: frames, payloads, telemetry, the CRC frames carry, what each bit of a header is; and whether decoded bits are as random as a
cipher's output"""
from . import frames, bits, payload, randomness, telemetry, crc, fields
