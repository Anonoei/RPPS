"""Sources, sinks, and the state carried from one block to the next

    python examples/streaming_io.py

Blocks come from a source (`rp.source.File`, `SigMF`, `Wav`, `Socket`, `Device` for a radio through pysdrlib,
`Memory` for an array or a capture held in memory) called with a block length. Stages added to the blocks run on
each one as it comes; `>>` writes them to a sink. Every block carries the stream's state, so a stage picks up exactly
where the last block left off:

    for out in src(4096) + rx: ...             # iterate
    src(4096) + rx >> rp.source.SigMFSink(..)  # or straight into a file
    src(4096).pipelined(rx)                    # each stage of rx in a thread of its own (real time)

Two kinds of state, carried for you when you iterate a source, by hand otherwise:
    a Stage (`+`) keeps it in the stream:  rp.Data(chunk, stream=prev.stream)
    a Pipe (`*`, `/`) keeps it in the context:  rp.Data(chunk, ctx=prev.ctx)
A fresh rp.Data(chunk) starts both over.
"""
import pathlib
import tempfile

import numpy as np

import rpps as rp

Fs, BAUD, FO = 1e6, 50e3, 150e3 # a 50 kBd QPSK carrier 150 kHz above the centre of a 1 MS/s recording

def recording(path):
    """Write a synthetic capture as a SigMF recording (16-bit integers, as a radio would give them)"""
    rng = np.random.default_rng(3)
    syms = rp.Data(rng.integers(0, 2, 2*20000).astype(bool), Fs=BAUD, sps=1) * rp.symbol.load("psk/qpsk")
    wave = syms + rp.wave.Shape(Fs/BAUD, rp.signal.filters.load("pulse/rrc/0p35"))
    air = wave + rp.signal.channel.CFO(FO + 900) + rp.signal.channel.AWGN(esn0_db=18, power=None, seed=1)
    x = (0.3*air.data/np.sqrt(np.mean(np.abs(air.data)**2))).astype(np.complex64)
    with rp.source.SigMFSink(path, datatype="ci16_le", description="streaming_io example") as sink:
        rp.source.Memory(x, Fs=Fs, cf=915e6)(8192) >> sink # Fs and cf come from the blocks
    return x

def conditioning():
    """Samples to the signal alone: shift it to 0 Hz, filter, decimate to 4 samples a symbol"""
    return (rp.signal.filters.ShiftFreq(-FO) + rp.signal.filters.low_pass(101, 60e3, Fs)
            + rp.signal.sample.down.Decimate(5))

def receiver():
    return conditioning() + rp.sync.Acquire() + rp.sync.time.Mat() + rp.sync.phase.FFCarrier() + rp.sync.quality.Monitor()

def main():
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp)/"capture"
        x = recording(path)
        # Fs, cf and the sample format come from the metadata. `with` unmaps the file when done: Windows can't delete
        # (or overwrite) a file that is still mapped
        with rp.source.SigMF(path, sps=Fs/BAUD) as src:
            print(f"{src}, {len(src)} samples")

            # 1. Iterate: rates follow the stages (Fs /5, sps from the detection), the detection rides on the data
            parts = []
            for out in src(4096) + receiver():
                parts.append(out.data)
        iterated = np.concatenate(parts)
        print(f"iterated: {len(iterated)} symbols at {out.Fs:.0f} Bd, {out.det.mod} detected, {out.stream.est['quality']}")

        # 2. By hand: the same blocks, the stream passed on
        rx, prev, parts = receiver(), None, []
        for at in range(0, len(x), 4096):
            blk = rp.Data(x[at:at + 4096], Fs=Fs, cf=915e6, sps=Fs/BAUD, flush=at + 4096 >= len(x),
                          stream=None if prev is None else prev.stream)
            prev = blk + rx
            parts.append(prev.data)
        by_hand = np.concatenate(parts)
        assert len(by_hand) == len(iterated) and np.allclose(by_hand, iterated, atol=2e-3) # (16-bit samples on disk)

        # 3. Pipelined: the source and every stage in a thread of their own, the same output; stream.load() is the time
        # each stage took as a fraction of real time (over 1.0: too slow for a live radio at this rate)
        parts = []
        with rp.source.SigMF(path, sps=Fs/BAUD) as src:
            for out in src(4096).pipelined(receiver()):
                parts.append(out.data)
        assert np.allclose(np.concatenate(parts), iterated)
        print("pipelined: same symbols; load per stage " + ", ".join(f"{type(s).__name__} {v:.2f}"
                                                                    for s, v in out.stream.load().items()))

        # 4. Any block cut gives the same samples through fixed stages (filters, shifts, resamplers). An estimator that
        # waits for enough signal (Acquire) can decide at a slightly different point, so compare before it
        with rp.source.SigMF(path) as src:
            a = (src(4096) + conditioning()).collect()
            b = (src(1500) + conditioning()).collect()
        assert np.allclose(a, b, atol=1e-5)
        print(f"conditioning in blocks of 4096 and 1500: the same {len(a)} samples")

        # 5. Pipes (`*`, `/`) carry their state in the context: decode symbols that arrive in pieces
        qpsk, conv = rp.symbol.load("psk/qpsk"), rp.code.load("conv/k7_171_133/1_2")
        msg = np.random.default_rng(4).integers(0, 2, 3000).astype(bool)
        syms = (rp.Data(msg, flush=True) * conv * qpsk).data
        prev, bits = None, []
        for at in range(0, len(syms), 777):
            d = rp.Data(syms[at:at + 777], sps=1, flush=at + 777 >= len(syms), ctx=None if prev is None else prev.ctx)
            prev = d / qpsk / conv
            bits.append(prev.as_bits().data)
        bits = np.concatenate(bits)
        assert np.array_equal(bits, msg[:len(bits)]) and len(bits) == len(msg)
        print(f"decoded {len(bits)} bits from symbols in blocks of 777, the decoder state carried by ctx=")

if __name__ == "__main__":
    main()
