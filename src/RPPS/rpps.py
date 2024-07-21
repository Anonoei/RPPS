from pyboiler.logger import Logger, Level

import cProfile
import pstats

Logger("RPPS", Level.TRACE)

class RPPS:
    def __init__(self):
        pass

def main():
    import mod
    from viz import show, DrawConstellation

    #pr = cProfile.Profile()
    #pr.enable()

    qpsk = mod.QPSK()
    # Encode data
    enc_str = b"This is a lot more binary data to send"
    print(f"Encoding!")
    r, meta = qpsk.generate(enc_str)
    print(f"Encoded {len(r)} symbols: \'{enc_str}\'")
    DrawConstellation(r, meta)
    show()
    meta.serialize(r)
    # Decode data
    print(f"Decoding!")
    data, meta = qpsk.process("qpsk.complex64.rpps.iq")
    print(f"Decoded {len(r)} symbols: \'{data}\'")
    #DrawConstellation(r, meta)
    #show()
    #pr.disable()
    #ps = pstats.Stats(pr).sort_stats(pstats.SortKey.CUMULATIVE)
    #ps.print_stats()

if __name__ == "__main__":
    main()
