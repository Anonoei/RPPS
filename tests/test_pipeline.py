import rpps as rp

def test_pipeline():
    mod = rp.mod.identify.by_name("QPSK", 0)
    ecc = rp.coding.Repetition(2)

    pipeline = rp.Pipeline(mod, ecc)

    enc_msg = b"""Test"""

    def encode(pipeline: rp.Pipeline, enc_msg):
        syms = pipeline.enc(enc_msg)

        # rp.viz.DrawConstellation(syms, meta)
        # mod.draw_refs()
        # rp.viz.show()

        return pipeline.meta.serialize(syms)

    def decode(pipeline: rp.Pipeline, path):
        data = pipeline.from_file(path)

    path = encode(pipeline, enc_msg)
    print()
    decode(pipeline, path)
