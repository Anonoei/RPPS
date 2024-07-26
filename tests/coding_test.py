import rpps as rp


def test_repetition():
    ecc = rp.coding.Repetition(4)

    enc_msg = b"""Hello world!"""

    enc = rp.Stream.from_bytes(enc_msg)

    encoded = ecc.encode(enc)

    decoded = ecc.decode(encoded)

    assert decoded.to_bytes() == enc_msg

if __name__ == "__main__":
    from utils import run_tests

    run_tests(globals())
