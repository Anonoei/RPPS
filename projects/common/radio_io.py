"""Shared capture helpers for the scanners and tools (wifi/, ble/, spectrum/, p25_dmr/, lte/, record.py, gain_search.py): open the radio, set its gain, grab samples, count clipping

    import radio_io as io
    sdr = io.open_radio(); io.start_rx(sdr)
    io.set_gain(sdr, 0.5); x = io.grab(sdr, n); io.clipped(x)

Works on any pysdrlib radio. Which one: `driver=`, else $RPPS_RADIO (a pysdrlib driver name), else the first radio pysdrlib finds.
With `remote=` (or PYSDRLIB_REMOTE=tcp://host:5555) the radio is the one a pysdrlib server hosts.

Gain is a fraction of the radio's own range, as pysdrlib's `set_rx_gain` takes it: 0.0 the least, 1.0 the most (linear in dB). The same fraction is the same place in the range on a
HackRF (which splits it over amp / LNA / VGA) and on a B210 (0-76 dB), so a ladder is a descent in gain on either. `describe(sdr)` prints what a fraction came to in dB.

A radio with two receive channels (B210) can use the second: with $RPPS_CHANNELS=2 `start_rx` streams both, and `capture_clean` and gain_search.search try two gains at once, one per channel.
That needs the same signal on both inputs (a splitter from one antenna to TX/RX and RX2; $RPPS_ANTENNA=name sets channel 0's input, and channel 1's is the other of TX/RX, RX2).

`capture_clean` implements the rule used for bursty signals: take the highest gain on a ladder that does not clip, checked over a window long enough to contain bursts.
"""
import os
import sys
import time

import numpy as np
import pysdrlib

LADDER = [round(1 - i/12, 4) for i in range(13)] # gain fractions, highest first: the descent the clipping searches take (about 7 dB a rung on a HackRF, 6 on a B210)
GAINS_FINE = [round(0.1*i, 1) for i in range(11)] # the steps a gain search that tries each one takes (a fraction of the range; 8 dB on a HackRF)
DEFAULT_GAIN = 0.4 # what a script that was not told a gain starts at (a HackRF: about 33 dB)

_driver = None

def _server_driver(remote):
    """The driver to ask a pysdrlib server for: the first type it hosts"""
    from pysdrlib.remote import Client
    with Client(remote, 5.0) as c:
        types = list(c.types())
    if not types:
        raise RuntimeError(f"the pysdrlib server at {remote} hosts no radio")
    return types[0]

def driver_name(driver=None, remote=None):
    """The pysdrlib driver to use: `driver`, else $RPPS_RADIO, else (for a remote radio) what the server has, else the first radio pysdrlib finds"""
    global _driver
    if driver or os.environ.get("RPPS_RADIO"):
        return driver or os.environ["RPPS_RADIO"]
    if remote:
        return _server_driver(remote)
    if _driver is None:
        try:
            found = [d.driver for d in pysdrlib.find()]
        except Exception: # a driver whose library is not built must not hide the others
            found = []
        if not found:
            raise RuntimeError("no radio found: plug one in, or set RPPS_RADIO to a pysdrlib driver name (or --remote / PYSDRLIB_REMOTE)")
        _driver = found[0]
    return _driver

def rate_limits(sdr=None, driver=None, remote=None):
    """(lowest, highest) sample rate in MS/s the radio supports, from pysdrlib's device caps (`caps.rates`). An open radio is asked directly; else the driver's class caps (no hardware needed).
    None for a bound the caps do not give."""
    try:
        caps = sdr.caps if sdr is not None else pysdrlib.device(driver_name(driver, remote or os.environ.get("PYSDRLIB_REMOTE"))).caps
        rates = list(caps.rates)
    except Exception: # no such driver here (a remote radio of a type this machine lacks) or no caps: no limit known
        rates = []
    if not rates:
        return None, None
    return min(r.lo for r in rates)/1e6, max(r.hi for r in rates)/1e6

def device(args=None, driver=None):
    """A local radio, not yet opened (see `driver_name`). `args` is unused (kept for scripts that pass their argparse namespace)"""
    return pysdrlib.device(driver_name(driver))

def add_gain_arg(ap, default=DEFAULT_GAIN):
    """`--gain` for an argparse parser: a fraction 0-1 of the radio's range"""
    ap.add_argument("--gain", type=float, default=default, help=f"receive gain as a fraction of the radio's range, 0-1 (default {default})")

def describe(sdr, channel=0):
    """The gain in effect as text: the fraction and what it came to in dB (a HackRF rounds to its steps)"""
    g = sdr.get_rx_gain(channel=channel)
    db = sdr.get_rx_gain_db(channel=channel)
    return f"{g:.2f}" if g is not None and db is None else f"{g:.2f} ({db:.0f} dB)"

def add_radio_args(ap):
    """`--radio DRIVER` and `--remote ADDRESS` for an argparse parser: which radio a script uses, so two scripts (or two radios of one script) can each have their own"""
    ap.add_argument("--radio", metavar="DRIVER", help="pysdrlib driver to use (default $RPPS_RADIO, else the first radio found)")
    ap.add_argument("--serial", help="open the radio with this serial number (a remote server lists them: `pysdrlib client ADDRESS list`)")
    ap.add_argument("--label", help="open a radio whose label contains this")
    ap.add_argument("--remote", metavar="ADDRESS", help="use a radio hosted by a pysdrlib server, e.g. tcp://HOST:5555 (default $PYSDRLIB_REMOTE, else a local radio)")

def channels_wanted():
    """1, or 2 when $RPPS_CHANNELS asks for both channels of a two-channel radio"""
    return int(os.environ.get("RPPS_CHANNELS", "1"))

def open_radio(cf=2437e6, fs=20e6, remote=None, driver=None, gain=DEFAULT_GAIN, program=None, serial=None, label=None, need=None, want=None):
    """Open a radio and tune it. program / serial / label / need / want say who this program is and what it needs, for a remote radio (pysdrlib >= 0.4.3): the server shows `program` next to the radio it has
    taken, `serial` / `label` pick a unit, and `need` (a radio that cannot is never used) / `want` (the nearest wins) are dicts of freq (Hz, or a (low, high) span), rate, bandwidth, channels, tx. With any of
    serial / label / need / want and no driver (`driver=`, $RPPS_RADIO) the server chooses the radio, instead of this taking the first type it hosts. A local radio ignores them. remote: a pysdrlib server address (tcp://host:5555, see `pysdrlib server`) to use a radio on another machine; defaults to the
    PYSDRLIB_REMOTE environment variable, so every script that opens its radio here can be pointed at a remote one without a flag. The returned object has the same
    methods either way (initialize, start_rx, read, set_rx_gain, close...). Start it with `start_rx(sdr)`."""
    remote = remote or os.environ.get("PYSDRLIB_REMOTE")
    if remote:
        choice = {k: v for k, v in dict(serial=serial, label=label, need=need, want=want).items() if v is not None}
        driver = driver or os.environ.get("RPPS_RADIO") or (None if choice else _server_driver(remote))
        sdr = pysdrlib.connect(remote, driver, reuse=True, program=program, **choice) # takes over a hosted radio of the type nobody controls, else opens one; already open
    else:
        sdr = device(driver=driver); sdr.open()
    sdr.initialize(cf=cf, Fs=fs, rx_gain=gain)
    got = sdr.get_rx_sample_rate()
    if got and abs(got - fs) > 1e-3*fs: # a radio that only has fixed rates (an RX-888) snaps to the nearest: the scripts' filters and timing are for `fs`, so a radio that cannot run at it is the wrong one
        sdr.close()
        raise RuntimeError(f"{getattr(sdr, 'id', None) or sdr.NAME} runs at {got/1e6:g} MS/s, not the {fs/1e6:g} requested: pick another radio (--radio DRIVER, --serial, --label)")
    antenna = os.environ.get("RPPS_ANTENNA")
    if antenna:
        sdr.set_rx_antenna(antenna)
    return sdr

def start_rx(sdr):
    """Start receiving: channel 0, or channels 0 and 1 when $RPPS_CHANNELS=2 and the radio has two (the second on the input the first is not)"""
    n = min(channels_wanted(), sdr.caps.channels.get("rx", 1) or 1)
    if n > 1:
        ants = list(sdr.rx_antennas())
        if len(ants) > 1:
            first = sdr.get_rx_antenna(0) or ants[0] # unknown until set: then the first listed
            sdr.set_rx_antenna(first, channel=0)
            sdr.set_rx_antenna(next(a for a in ants if a != first), channel=1)
        sdr.start_rx(channels=tuple(range(n)))
    else:
        sdr.start_rx()

def dual(sdr):
    """True when the receive stream carries two channels"""
    return len(sdr.rx_channels) > 1

def set_gain(sdr, gain, channel=0):
    """Set a channel's receive gain, a fraction 0-1 of the radio's range"""
    sdr.set_rx_gain(gain, **({"channel": channel} if channel else {}))

def gain_db(sdr, channel=0):
    """The gain in effect on a channel in dB, the total over its stages (0.0 if the radio does not say)"""
    db = sdr.get_rx_gain_db(channel=channel)
    return float(db) if db is not None else 0.0

def grab(sdr, n):
    """`n` samples (channel 0) from the running stream, starting now. Lost samples are reported by pysdrlib itself (`pysdrlib.warn.Dropped`, counts in `sdr.stats`)"""
    if dual(sdr):
        return grab2(sdr, n)[0]
    sdr.get_samples() # drop what queued while we were busy
    out, k = [], 0
    while k < n:
        b = sdr.get_samples()
        if len(b):
            out.append(b); k += len(b)
        else:
            time.sleep(0.001)
    return np.concatenate(out)[:n]

def grab2(sdr, n):
    """`n` samples of both channels, the same instants, starting now: a (2, n) array"""
    chs = sdr.rx_channels
    for ch in chs:
        sdr.read_available(channel=ch) # drop what queued while we were busy
    return sdr.read(n, channel=chs)

def clipped(x):
    return float(np.mean((np.abs(x.real) > 0.98) | (np.abs(x.imag) > 0.98)))

def capture_clean(sdr, fs, seconds, check_seconds=0.3, limit=2e-5, final_limit=1e-4, ladder=LADDER):
    """Highest ladder rung (gain fraction) whose clipping over `check_seconds` is below `limit`; capture `seconds`; if the capture itself clipped more than `final_limit`
    (a burst the check window missed) step down and capture again. Returns (samples, gain fraction, clipped fraction).
    With two channels streaming (see `start_rx`) two rungs are checked at a time, one on each channel, so the search takes half as many windows."""
    two = dual(sdr)
    start = 0
    for _ in range(len(ladder)):
        i = start
        while i < len(ladder):
            if two:
                j = min(i + 1, len(ladder) - 1)
                set_gain(sdr, ladder[i]); set_gain(sdr, ladder[j], channel=1); time.sleep(0.03)
                w = grab2(sdr, int(fs*check_seconds))
                if clipped(w[0]) < limit:
                    break
                if clipped(w[1]) < limit:
                    i = j
                    break
                i += 2
            else:
                set_gain(sdr, ladder[i]); time.sleep(0.03)
                if clipped(grab(sdr, int(fs*check_seconds))) < limit:
                    break
                i += 1
        start = min(i, len(ladder) - 1)
        if two:
            set_gain(sdr, ladder[start]); time.sleep(0.03)
        x = grab(sdr, int(fs*seconds))
        c = clipped(x)
        if c < final_limit or start == len(ladder) - 1:
            return x, ladder[start], c
        start += 1
    return x, ladder[start], c
