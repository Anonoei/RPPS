import rpps as rp


def test_modulation():
    mod = rp.mod.identify.by_name("QPSK", 1)

    enc_msg = b"""Hello world!"""

    enc = rp.Stream.from_bytes(enc_msg)
    symbols, meta = mod.modulate(enc)

    meta = rp.Meta.from_name(meta.to_name())
    data, meta = mod.demodulate(symbols, meta)

    assert data.to_bytes() == enc_msg

def test_serialization():
    mod = rp.mod.identify.by_name("QPSK", 1)

    enc_msg = b"""Hello world!"""

    enc = rp.Stream.from_bytes(enc_msg)
    symbols, meta = mod.modulate(enc)

    path = meta.serialize(symbols)

    meta, symbols = rp.Meta.from_file(path)

    data, meta = mod.demodulate(symbols, meta)

    assert data.to_bytes() == enc_msg


if __name__ == "__main__":
    from utils import run_tests

    run_tests(globals())
