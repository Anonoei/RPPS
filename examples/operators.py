"""The data model: `*` encodes, `/` decodes, and the data carries what it is

    python examples/operators.py

RPPS has no graph to build and nothing to reset. Operators are applied to an `rp.Data`, which knows its representation
(samples, symbols, channel bits, data bits, packets), its rates, and the state of every operator that has touched it:

    data * op   encode / modulate / scramble (transmit direction)
    data / op   decode / demodulate / descramble (receive direction)
    data + op   a DSP stage (samples to samples, see channel_and_sync.py)

Operators multiplied together are a reusable link that decodes in the opposite order. The representation is checked at
every step, so a wrong order is an error instead of garbage.
"""
import numpy as np

import rpps as rp

def main():
    # Definitions ship with the package: `python -m rpps ls` lists them
    mod = rp.symbol.load("psk/qpsk")   # a constellation definition (rp.symbol.generate("16QAM") makes a shape)
    ecc = rp.code.Hamming(7, 4)        # or rp.code.load("block/hamming/7_4")
    scr = rp.code.scram.load("v35")

    # One step at a time
    msg = rp.Data(b"Hello World!")      # bytes are bytes; a bool array is bits
    syms = msg * scr * ecc * mod        # scramble, encode, modulate
    out = syms / mod / ecc / scr        # demodulate, decode, descramble: the reverse order
    print(f"{syms} -> {out}")
    assert out.as_bytes().hex == msg.as_bytes().hex

    # The same thing as one link: multiply the operators once, use it both ways
    link = scr * ecc * mod
    print(f"link: {link}")
    assert (rp.Data(b"Hello World!") * link / link).as_bytes().hex == msg.as_bytes().hex

    # Partial blocks wait for more. QPSK takes 2 bits a symbol, so 3 bits give 1 symbol and the 3rd bit waits in the
    # data's context for the next call; flush=True says there is no more and pads it out
    bits = np.array([1, 0, 1], dtype=bool) # bits are bool: an int 0/1 array is an error, not bits
    print(f"3 bits -> {len(rp.Data(bits) * mod)} symbol, flushed -> {len(rp.Data(bits, flush=True) * mod)}")
    assert len(rp.Data(bits) * mod) == 1 and len(rp.Data(bits, flush=True) * mod) == 2

    # A link layer works on data bits, not channel bits. With no FEC, rp.code.Uncoded() says "these channel bits are the
    # data bits": `syms / mod / crc` is refused, because the representations don't line up
    crc = rp.link.load("crc/crc16_xmodem")
    unc = rp.code.Uncoded()
    framed = msg * crc * unc * mod
    back = framed / mod / unc / crc
    print(f"crc ok: {crc.encoder.ok}, {len(framed)} symbols for {len(msg.as_bytes().hex)//2} bytes + 16 CRC bits")
    assert crc.encoder.ok and back.as_bytes().hex == msg.as_bytes().hex
    try:
        _ = framed / mod / crc
        raise AssertionError("channel bits went into a CRC")
    except rp.errors.StreamError as e:
        print(f"refused, as it should be: {e}")

    # The demodulator's output is soft: LLRs, log(P0/P1), positive for a 0 (see soft_decisions.py)
    # (noiseless symbols give enormous ones, so add a little noise first)
    llr = (syms + rp.signal.channel.AWGN(esn0_db=6, seed=1)) / mod
    print(f"demodulated: {llr}, soft={llr.is_soft}, first LLRs {np.round(llr.get_soft()[:4], 1)}")

if __name__ == "__main__":
    main()
