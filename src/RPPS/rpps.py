from pyboiler.logger import Logger, Level

import time
import cProfile
import pstats

Logger("RPPS", Level.TRACE)

class RPPS:
    def __init__(self):
        pass

def main():
    import mod
    import fec

    from helpers.encoding import Encoding
    from viz import show, DrawConstellation

    #pr = cProfile.Profile()
    #pr.enable()

    enc = Encoding.from_bytes(b"""
        Hello world! This is a much longer message containing way more bytes for this to process so I can test how well it performs
        Hello world! This is a much longer message containing way more bytes for this to process so I can test how well it performs
        Hello world! This is a much longer message containing way more bytes for this to process so I can test how well it performs
        Hello world! This is a much longer message containing way more bytes for this to process so I can test how well it performs
        Hello world! This is a much longer message containing way more bytes for this to process so I can test how well it performs
        Hello world! This is a much longer message containing way more bytes for this to process so I can test how well it performs
        Hello world! This is a much longer message containing way more bytes for this to process so I can test how well it performs
        Hello world! This is a much longer message containing way more bytes for this to process so I can test how well it performs
        Hello world! This is a much longer message containing way more bytes for this to process so I can test how well it performs
        Hello world! This is a much longer message containing way more bytes for this to process so I can test how well it performs
        """
    )
    #print(f"Raw: {enc.bitarray}")

    # Add FEC
    #vit = fec.Viterbi(3, 4)
    #encoded, trellis = vit.encode(enc)
    #print(f"Encoded: {encoded}")
    #print(f"Trellis: {trellis}")
    #enc = Encoding.from_bytes(encoded.to_bytes())

    length = len(enc.bytes)
    qpsk = mod.QPSK()


    print(f"Coding {length} bytes each itteration")

    time_start = time.perf_counter()
    ittrs = 20
    for _ in range(ittrs):
        # Encode data
        #print(f"Encoding!")
        r, meta = qpsk.generate(enc)
        #print(f"Encoded {len(r)} symbols: \'{enc.bytes}\'")
        #DrawConstellation(r, meta)
        #show()
        #meta.serialize(r)
    time_dur = time.perf_counter() - time_start
    print(f"Encode x{ittrs} took {time_dur:.2f}s, avg: {time_dur/ittrs:.2f}")

    time_start = time.perf_counter()
    for _ in range(ittrs):
        # Decode data
        #print(f"Decoding!")
        data, meta = qpsk.process("qpsk.complex64.rpps.iq")
        #print(f"Decoded {len(r)} symbols: \'{data.bytes}\'")

        #decoded = vit.decode(data)
        #print(f"Decoded: {decoded}")
        #DrawConstellation(r, meta)
        #show()
    time_dur = time.perf_counter() - time_start
    print(f"Decode x{ittrs} took {time_dur:.2f}s, avg: {time_dur/ittrs:.2f}")
    #pr.disable()
    #ps = pstats.Stats(pr).sort_stats(pstats.SortKey.CUMULATIVE)
    #ps.print_stats()

if __name__ == "__main__":
    main()
