from pyboiler.logger import Logger, Level

import time
import cProfile
import pstats

import rpps as rp

def main():

    ecc = rp.coding.Repetition(3, 4)
    mod = rp.mod.QPSK(1)

    enc_msg = b"""Hello world!"""

    def encode(ecc, mod, enc_msg):
        time_start = time.perf_counter()
        print("Encoding!")
        enc = rp.Stream.from_bytes(enc_msg)
        print(f"Encoding {enc.bytes}")

        encoded = ecc.encode(enc)

        enc = rp.Stream.from_bytes(encoded.to_bytes())

        r, meta = mod.modulate(enc)

        print(f"Modulated {len(r)} symbols")
        time_dur = time.perf_counter() - time_start
        print(f"Encode took {time_dur:.2f}s")

        # DrawConstellation(r, meta)
        # show()

        return meta.serialize(r)

    def decode(ecc, mod, path):
        time_start = time.perf_counter()
        print("Decoding!")
        meta, symbols = rp.Meta.from_file(path)

        data, meta = mod.demodulate(symbols, meta)
        print(f"Demodulated {len(symbols)} symbols")

        decoded = ecc.decode(data.bitarray)
        print(f"Decoded: {decoded.to_bytes()}")

        time_dur = time.perf_counter() - time_start
        print(f"Decode took {time_dur:.2f}s")

        # DrawConstellation(r, meta)
        # show()

    path = encode(ecc, mod, enc_msg)
    print()
    decode(ecc, mod, path)


if __name__ == "__main__":
    # pr = cProfile.Profile()
    # pr.enable()
    main()
    # pr.disable()
    # ps = pstats.Stats(pr).sort_stats(pstats.SortKey.CUMULATIVE)
    # ps.print_stats()
