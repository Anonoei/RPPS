"""Stages must give the same stream no matter how it's split into blocks"""
import numpy as np
import rpps as rp

def run(stage, x, blocks, **meta_args):
    meta = rp.Meta(**meta_args)
    outs, i = [], 0
    for b in blocks:
        meta.obj = x[i:i+b]
        i += b
        meta = meta + stage
        outs.append(meta.obj)
        meta.reset()
    return np.concatenate(outs)

def check(name, make, x, **meta_args):
    splits = ([len(x)], [1000]*(len(x)//1000), [777]*(len(x)//777) + [len(x) % 777], [1]*50 + [len(x)-50])
    ref = run(make(), x, splits[0], **meta_args)
    for sp in splits[1:]:
        y = run(make(), x, sp, **meta_args)
        assert len(y) == len(ref), f"{name}: {len(y)} != {len(ref)} samples for blocks {sp[:2]}.."
        assert np.allclose(y, ref, atol=1e-6), f"{name}: output depends on block size {sp[:2]}.."
    print(f"{name}: ok ({len(ref)} samples)")

def main():
    rng = np.random.default_rng(0)
    x = (rng.standard_normal(40000) + 1j*rng.standard_normal(40000)).astype(np.complex64)
    taps = rp.filters.designer.designer(101, 7.5e3, fs=250e3)
    check("FMDemod", lambda: rp.mod.FMDemod(), x, Fs=250e3)
    check("ShiftFreq", lambda: rp.filters.ShiftFreq(-57e3), x, Fs=250e3)
    check("FIR", lambda: rp.filters.filters.ConvFilter(taps), x, Fs=250e3)
    check("Decimate", lambda: rp.sample.down.Decimate(10), x, Fs=250e3, sps=10)
    check("Polyphase", lambda: rp.sample.re.Polyphase(19, 25), x, Fs=25e3, sps=25)
    check("Biphase", lambda: rp.filters.ShapingBiphase(), x, Fs=19e3, sps=16)

if __name__ == "__main__":
    main()
