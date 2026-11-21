"""A known signal from one definition: a profile transmits and receives it

    python examples/profile_loopback.py
    python -m rpps ls profile          # the shipped profiles
    python -m rpps show profile/ble/adv
    python -m rpps steps               # the step kinds and sync methods a profile can name

A profile is one JSON file with the steps of a signal in receive order. `p.rx()` runs them as written; `p.tx()` runs the
transform steps inverted and in reverse (sync steps are receive only). Header fields go in on transmit (`fields=`) and
come back on receive in `stream.est`. When a signal has a profile, use it instead of chaining operators by hand: the
header-driven changes (lengths, rates, modulations) are wired up for you.

Here: Bluetooth LE advertising packets on channel 38 through a frequency offset and noise. BLE's profile receives packet
by packet: its burst step cuts the samples into packets, and the steps after it run on each packet anew (its preamble
step finds the access address, which places the bit timing and takes off the frequency offset). So a runner gives a list,
one output per packet that ended in the block, each with its own `stream.est`. A runner goes where a stage does,
`source(n) + p.rx()` (blocks flatten the lists), and how the samples are cut doesn't matter: the receive is run in blocks
of two sizes, and once more by hand in uneven blocks (`stream=` passed on, as with any stage).
"""
import numpy as np

import rpps as rp

def main(snr_db=15.0, cfo=20e3):
    rng = np.random.default_rng(11)
    p = rp.profile.load("ble/adv", channel=38)
    print(p, p.params)

    # Transmit three advertising PDUs, each a burst of its own (flush=True: its last pulses go out whole), 1 ms apart
    pdus = [bytes(rng.integers(0, 256, n).astype(np.uint8)) for n in (6, 20, 37)]
    tx, stream, waves = p.tx(), None, [np.zeros(36, np.complex64)]
    for pdu in pdus:
        d = rp.Data(pdu, Fs=1e6, sps=1, stream=stream, flush=True)
        stream = d.stream
        wave = tx(d, fields={"type": 2, "txadd": 1}) # header fields; the length field is worked out
        waves += [wave.data, np.zeros(4000, np.complex64)]
    air = (rp.Data(np.concatenate(waves), Fs=wave.Fs)
           + rp.signal.channel.CFO(cfo) + rp.signal.channel.AWGN(snr_db=snr_db, seed=1))
    print(f"transmitted {len(pdus)} packets: {len(air)} samples at {wave.Fs/1e6:g} MS/s, {wave.sps} a bit")

    # Receive: the samples as a source gives them, the runner added like a stage; one output per packet
    def receive(block):
        outs = list(rp.source.Memory(air.data, Fs=wave.Fs, sps=wave.sps)(block) + p.rx())
        return b"".join(bytes.fromhex(o.as_bytes().hex) for o in outs if len(o)), outs
    got, outs = receive(4096)
    for o in outs:
        est = o.stream.est
        print(f"packet at sample {est['aa_wave']['pos']}: header {est['ble_adv']['last']}, offset level {est['aa_wave']['level']:+.3f}")
    assert got == b"".join(pdus)
    assert len(outs) == 3 and all(o.stream.est["ble_adv"]["fcs_failed"] == 0 for o in outs)
    assert outs[-1].stream.est["ble_adv"]["last"]["txadd"] == 1
    assert receive(300)[0] == got

    # By hand, in uneven blocks: the runner called on each, the stream passed on
    rx, stream, parts, at = p.rx(), None, [], 0
    cuts = [333, 1, 700, 5000]
    cuts.append(len(air) - sum(cuts))
    for k, n in enumerate(cuts):
        blk = rp.Data(air.data[at:at + n], Fs=wave.Fs, sps=wave.sps, stream=stream, flush=k == len(cuts) - 1)
        stream, at = blk.stream, at + n
        parts += [bytes.fromhex(o.as_bytes().hex) for o in blk + rx]
    assert b"".join(parts) == got
    print(f"the same {len(got)} bytes in blocks of 4096, of 300, and of {cuts}")

if __name__ == "__main__":
    main()
