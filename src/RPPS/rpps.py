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
    import coding

    from helpers.stream import Stream
    from viz import show, DrawConstellation

    pr = cProfile.Profile()
    pr.enable()


    ecc = coding.Repetition(3, 4)
    qpsk = mod.QPSK(1)

    time_start = time.perf_counter()
    ittrs = 100
    for _ in range(ittrs):
        # Encode data
        #print(f"Encoding!")
        enc = Stream.from_bytes(b"""Hello world!""")
        encoded = ecc.encode(enc)
        enc = Stream.from_bytes(encoded.to_bytes())
        r, meta = qpsk.modulate(enc)
        #print(f"Modulated {len(r)} symbols: \'{enc.bytes}\'")
        #DrawConstellation(r, meta)
        #show()
        meta.serialize(r)
    time_dur = time.perf_counter() - time_start
    print(f"Encode x{ittrs} took {time_dur:.2f}s, avg: {time_dur/ittrs:.2f}")

    time_start = time.perf_counter()
    for _ in range(ittrs):
        # Decode data
        #print(f"Decoding!")
        data, meta = qpsk.demodulate("qpsk.complex64.rpps.iq")
        #print(f"Demodulated {len(r)} symbols: \'{data.bytes}\'")

        decoded = ecc.decode(data.bitarray)
        #print(f"Decoded: {decoded.to_bytes()}")
        #DrawConstellation(r, meta)
        #show()
    time_dur = time.perf_counter() - time_start
    print(f"Decode x{ittrs} took {time_dur:.2f}s, avg: {time_dur/ittrs:.2f}")
    pr.disable()
    ps = pstats.Stats(pr).sort_stats(pstats.SortKey.CUMULATIVE)
    ps.print_stats()

if __name__ == "__main__":
    main()
