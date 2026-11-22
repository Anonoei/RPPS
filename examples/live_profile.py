"""Profile the signals of a live stream: capture from a pysdrlib source, find every signal, pick some, and print each one's
profile (rp.analysis.profile: every step that processes it, and what it might be where that is unknown)

    python examples/live_profile.py --tcpiq HOST:PORT --format int16 --cf 1.2e9 --fs 25e6     # raw I/Q over TCP
    python examples/live_profile.py --radio hackrf --cf 851e6 --fs 10e6 --gain 0.6             # a radio here
    python examples/live_profile.py --remote tcp://HOST:5555 --cf 851e6 --fs 10e6              # one a pysdrlib server hosts

The stream is captured into memory first (`--seconds`; the analysis runs slower than the radio, which would drop samples
under it), surveyed, and the signals listed. `--pick 0,3` (or `all`) chooses which to profile; without it you are asked.
`--loop` captures and profiles again and again, following the chosen signals by frequency, until Ctrl-C.
"""
import argparse
import sys

import numpy as np
import pysdrlib
from pysdrlib.base import convert

import rpps as rp

FORMATS = {"int8": convert.INT8, "uint8": convert.UINT8, "int16": convert.INT16, "int12": convert.INT12,
           "float32": convert.FLOAT32}

def source(args):
    """The rp.source.Device the arguments name"""
    if args.tcpiq:
        host, _, port = args.tcpiq.rpartition(":")
        dev = pysdrlib.device("tcpiq", FORMATS[args.format])
        return rp.source.Device(dev, cf=args.cf, Fs=args.fs, host=host, port=int(port))
    if args.remote: # The server picks a radio that can tune and sample as asked (or the --radio kind)
        sdr = pysdrlib.connect(args.remote, args.radio, reuse=True, program="rpps live_profile",
                               need={"freq": args.cf, "rate": args.fs})
        return rp.source.Device(sdr, cf=args.cf, Fs=args.fs, rx_gain=args.gain)
    if args.radio is None:
        found = [d.driver for d in pysdrlib.find()]
        if not found:
            sys.exit("no radio found: give --radio, --remote or --tcpiq")
        args.radio = found[0]
    return rp.source.Device(args.radio, cf=args.cf, Fs=args.fs, rx_gain=args.gain)

def choose(survey, pick):
    """Indices of the signals to profile: `pick` ("all", "0,3"), else asked (all when there is no one to ask)"""
    if not len(survey):
        return []
    if pick is None and sys.stdin.isatty():
        pick = input(f"profile which signals (0-{len(survey) - 1}, comma separated, or all) [all]: ").strip() or "all"
    if pick is None or pick == "all":
        return list(range(len(survey)))
    idx = [int(x) for x in pick.replace(" ", ",").split(",") if x != ""]
    bad = [i for i in idx if not 0 <= i < len(survey)]
    if bad:
        sys.exit(f"no signal {bad}: there are {len(survey)}")
    return idx

def follow(survey, freqs):
    """Indices of the signals nearest the frequencies picked the first time (a signal may have moved a little)"""
    out = []
    for f in freqs:
        if len(survey):
            i = int(np.argmin([abs(s.freq - f) for s in survey]))
            if abs(survey[i].freq - f) < survey[i].width and i not in out:
                out.append(i)
    return out

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_argument_group("source (one of)")
    src.add_argument("--tcpiq", metavar="HOST:PORT", help="raw interleaved I/Q from a TCP server")
    src.add_argument("--remote", metavar="ADDRESS", help="a radio a pysdrlib server hosts (tcp://host:5555)")
    src.add_argument("--radio", metavar="DRIVER", help="pysdrlib driver (default: the first radio found / the server hosts)")
    ap.add_argument("--format", default="int16", choices=sorted(FORMATS), help="--tcpiq sample format (default int16)")
    ap.add_argument("--cf", type=float, required=True, help="centre frequency, Hz (for --tcpiq: what the stream is)")
    ap.add_argument("--fs", type=float, required=True, help="sample rate, samples/s")
    ap.add_argument("--gain", type=float, default=None, help="receive gain, a fraction 0-1 of the radio's range")
    ap.add_argument("--seconds", type=float, default=None,
                    help="how much to capture (default: what the profile reads, --blocks x 16384 samples)")
    ap.add_argument("--blocks", type=int, default=450, help="blocks of 16384 samples each signal's chain runs over")
    ap.add_argument("--settle", type=float, default=0.5,
                    help="seconds thrown away before each capture (a radio just tuned delivers the old frequency for a while)")
    ap.add_argument("--pick", help="signals to profile: all, or indices (0,3); default: ask")
    ap.add_argument("--no-blind", action="store_true", help="don't run the blind tools where no definition fits (faster)")
    ap.add_argument("--loop", action="store_true", help="capture and profile again until Ctrl-C")
    args = ap.parse_args()

    dev = source(args)
    count = int(args.seconds*args.fs) if args.seconds else args.blocks*16384
    freqs = None
    try:
        while True:
            mem = rp.source.Memory.capture(dev, count, skip=int(args.settle*dev.Fs))
            print(f"\n{mem}" + (f", {mem.dropped} samples dropped by the source" if mem.dropped else ""))
            survey = rp.analysis.survey(mem)
            for i, s in enumerate(survey):
                print(f"  {i:3d}: {s.describe()}")
            idx = choose(survey, args.pick) if freqs is None else follow(survey, freqs)
            if freqs is None:
                freqs = [survey[i].freq for i in idx]
            if not idx:
                print("nothing to profile")
            else:
                print(f"profiling {len(idx)} signal{'s' if len(idx) > 1 else ''}" + ("" if args.no_blind else
                      " (blind tools on: tens of seconds each where no definition fits)") + "\n")
                for p in rp.analysis.profile(mem, signals=idx, survey=survey, blocks=args.blocks, blind=not args.no_blind):
                    print(p, "\n")
            if not args.loop:
                break
    except KeyboardInterrupt:
        print("stopped")

if __name__ == "__main__":
    main()
