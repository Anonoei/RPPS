"""A known signal from one definition: a profile transmits and receives it

    python examples/profile_loopback.py
    python -m rpps ls profile          # the shipped profiles
    python -m rpps show profile/ble/adv
    python -m rpps steps               # the step kinds and sync methods a profile can name

A profile is one JSON file with the steps of a signal in receive order. `p.rx()` runs them as written; `p.tx()` runs the
transform steps inverted and in reverse (sync steps are receive only). Header fields go in on transmit (`fields=`) and
come back on receive in `stream.est`. When a signal has a profile, use it instead of chaining operators by hand: the
header-driven changes (lengths, rates, modulations) are wired up for you.

Here: Bluetooth LE advertising packets on channel 38 through a frequency offset and noise. A runner goes where a stage
does, `source(n) + p.rx()`, and carries its state from block to block in the stream, so how the samples are cut
doesn't matter: the receive is run in blocks of two sizes, and once more by hand in uneven blocks (`stream=` passed
on, as with any stage).

`sync=` overrides a profile's sync methods for one load. BLE's timing is estimated on the stream (Oerder & Meyr), which
wants a continuous signal; these are packets with silence between them, where noise would drive the estimate, and
per-burst acquisition is not built yet (README roadmap), so this takes timing as known: the transmitter's.
"""
import numpy as np

import rpps as rp

def main(snr_db=15.0, cfo=20e3):
    rng = np.random.default_rng(11)
    p = rp.profile.load("ble/adv", channel=38, sync={"timing": {"method": "known"}})
    print(p, p.params)

    # Transmit three advertising PDUs, each a burst of its own (flush=True: its last pulses go out whole)
    pdus = [bytes(rng.integers(0, 256, n).astype(np.uint8)) for n in (6, 20, 37)]
    tx, stream, waves = p.tx(), None, [np.zeros(36, np.complex64)]
    for pdu in pdus:
        d = rp.Data(pdu, Fs=1e6, sps=1, stream=stream, flush=True)
        stream = d.stream
        wave = tx(d, fields={"type": 2, "txadd": 1}) # header fields; the length field is worked out
        waves += [wave.data, np.zeros(68, np.complex64)]
    air = (rp.Data(np.concatenate(waves), Fs=wave.Fs)
           + rp.signal.channel.CFO(cfo) + rp.signal.channel.AWGN(snr_db=snr_db, seed=1))
    print(f"transmitted {len(pdus)} packets: {len(air)} samples at {wave.Fs/1e6:g} MS/s, {wave.sps} a bit")

    # Receive: the samples as a source gives them, the runner added like a stage
    def receive(block):
        got, stream = [], None
        for out in rp.source.Memory(air.data, Fs=wave.Fs, sps=wave.sps)(block) + p.rx():
            stream = out.stream
            if len(out): # a packet comes out of the block its CRC is checked in
                got.append(bytes.fromhex(out.as_bytes().hex))
        return b"".join(got), stream
    got, stream = receive(4096)
    est = stream.est["ble_adv"]
    print(f"received {est['frames']} packets, {est['fcs_failed']} failed the CRC; last header {est['last']}")
    print(f"access address seen {stream.est['aa']['frames']} times")
    assert got == b"".join(pdus)
    assert est["frames"] == 3 and est["fcs_failed"] == 0 and est["last"]["txadd"] == 1
    assert receive(300)[0] == got

    # By hand, in uneven blocks: the runner called on each, the stream passed on
    rx, stream, parts, at = p.rx(), None, [], 0
    cuts = [333, 1, 700]
    cuts.append(len(air) - sum(cuts))
    for k, n in enumerate(cuts):
        blk = rp.Data(air.data[at:at + n], Fs=wave.Fs, sps=wave.sps, stream=stream, flush=k == len(cuts) - 1)
        stream, at = blk.stream, at + n
        parts.append(bytes.fromhex((blk + rx).as_bytes().hex))
    assert b"".join(parts) == got
    print(f"the same {len(got)} bytes in blocks of 4096, of 300, and of {cuts}")

if __name__ == "__main__":
    main()
