import rpps as rp


def test_modulation():
    qpsk = rp.mod.QPSK(1)

    enc_msg = b"""Hello world!"""

    enc = rp.Stream.from_bytes(enc_msg)
    symbols, meta = qpsk.modulate(enc)

    meta = rp.Meta.from_name(meta.to_name())
    data, meta = qpsk.demodulate(symbols, meta)

    assert data.to_bytes() == enc_msg

def test_serialization():
    qpsk = rp.mod.QPSK(1)

    enc_msg = b"""Hello world!"""

    enc = rp.Stream.from_bytes(enc_msg)
    symbols, meta = qpsk.modulate(enc)

    path = meta.serialize(symbols)

    meta, symbols = rp.Meta.from_file(path)

    data, meta = qpsk.demodulate(symbols, meta)

    assert data.to_bytes() == enc_msg


if __name__ == "__main__":
    from utils import run_tests

    run_tests(globals())
