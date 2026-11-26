"""
RPPS command line interface

    python -m rpps ls [category ...] [--tag dvb-s2] [--tags] [--all]
    python -m rpps steps
    python -m rpps survey <file> --fs <Hz> [--fmt <format>] [--cf <Hz>] [--plot]
    python -m rpps profile <file> --fs <Hz> [--fmt <format>] [--signals 0,3] [--no-blind] [--json out.jsonl]
    python -m rpps solve <bits or symbols file> [--format bits|bytes|npy] [--mod qpsk]
    python -m rpps fmt [--check] [dir]
    python -m rpps gen-ldpc <dvb_s2_tables.hh> [table ids ...] [--out dir]
"""
import argparse
import json
import re
import shutil
import sys
import textwrap
from pathlib import Path


# name -> (title, function of the rpps module giving {group: [names]}). Static, so --help needs no import of rpps
CATEGORIES = {
    "constellation": ('Constellations  (rp.symbol.load("<group>/<name>"[, "<mapping>"]); shapes: rp.symbol.generate("16QAM"))', lambda rp: rp.symbol.ls()),
    "pulse": ('Pulses  (rp.signal.filters.load("pulse/<family>/<name>"))', lambda rp: rp._config.ls("pulse")),
    "filter": ('Filters  (rp.signal.filters.load("<kind>/<family>/<name>"))', lambda rp: rp._config.ls("filter")),
    "sequence": ('Sequences  (rp.sequence.load("<name>"))', lambda rp: rp.sequence.ls()),
    "scramble": ('Scramblers  (rp.code.scram.load("<name>"))', lambda rp: rp.code.scram.ls()),
    "fec": ('FEC  (rp.code.load("<kind>/<family>/<name>"))', lambda rp: rp._config.ls("fec")),
    "interleave": ('Interleavers  (rp.code.load("interleave/<type>/<name>"))', lambda rp: rp._config.ls("interleave")),
    "crc": ('CRCs  (rp.link.load("crc/<name>"))', lambda rp: rp._config.ls("crc")),
    "frame": ('Frames  (rp.link.load("frame/<name>"))', lambda rp: rp._config.ls("frame")),
    "header": ('Headers  (rp.symbol.load_header("<name>"), rp.link.load("header/<name>"))', lambda rp: rp._config.ls("header")),
    "table": ('Tables  (rp.profile.table("<family>/<name>"))', lambda rp: rp._config.ls("table")),
    "profile": ('Profiles  (rp.profile.load("<family>/<mode>"))', lambda rp: rp.profile.ls()),
    "plan": ('Burst time plans  (rp.plan.load("<family>/<name>"); builders, "base": "builder": rp.plan.builder("<family>/<name>"))', lambda rp: rp.plan.ls()),
    "linktype": ("Link types  (rp.link.linktype(<name>): packets for Wireshark and interfaces, a profile's \"capture\")",
                 lambda rp: rp._config.ls("linktype")),
    "system": ('Systems  (rp.analysis.catalogue.systems(): what catalogue.match scores against)', lambda rp: rp._config.ls("system")),
    "windows": ("Built-in Windows  (rp.signal.filters.Window.get(<name>))", lambda rp: {"windows": rp.signal.filters.Window.s()}),
}


def _show(title, groups, width):
    """Print a title, then one wrapped, aligned line per group"""
    print(f"\n{title}")
    if not groups:
        print("  (none)")
        return
    pad = max(len(k) for k in groups)
    for k, names in groups.items():
        lead = f"  {k:<{pad}}  "
        print(textwrap.fill(", ".join(names), width - len(lead), initial_indent=lead, subsequent_indent=" " * len(lead)))


def cmd_ls(args):
    import rpps as rp
    width = min(shutil.get_terminal_size((100, 20)).columns, 120)
    if not rp._config._SHIPPED:
        print(f"Only the user's definitions ($RPPS_CONFIG_SHIPPED=0): {', '.join(str(r) for r in rp._config.roots()) or 'none'}")
    if rp._config.selected_tags():
        print(f"The blind tools try only tags {', '.join(rp._config.selected_tags())} and untagged definitions ($RPPS_CONFIG_TAGS)")
    if args.tags:
        used = rp.tagged()
        _show("Tags  (python -m rpps ls --tag <tag>)", {t: [f"{v.get('comment', '')} ({sum(len(r) for r in used.get(t, {}).values())})"]
                                                      for t, v in rp.tags().items()}, width)
        return 0
    if args.tag:
        _show(f"Tagged {args.tag}  (rp.tagged({args.tag!r}))", rp.tagged(args.tag, all=args.all), width)
        return 0
    if args.all or args.blind:
        for name in args.category or [c for c in CATEGORIES if c in rp._config.domains()]:
            _show(name, rp._config.ls(name, all=args.all, blind=args.blind), width)
        return 0
    for name in args.category or CATEGORIES:
        title, fn = CATEGORIES[name]
        _show(title, fn(rp), width)
    return 0


def cmd_steps(args):
    """The step kinds a profile's "steps" can name, by category, and the methods of the sync kinds that have them"""
    from rpps.profile import steps
    width = min(shutil.get_terminal_size((100, 20)).columns, 120)
    print("A profile step is {\"<category>\": \"<kind>\", ...}, in receive order (python -m rpps show profile/<family>/<mode> for examples):")
    _show("Step kinds", steps.kinds(), width)
    _show("Sync methods  (\"method\" on the step, or \"defaults\": {\"timing\": {\"method\": ...}})",
          {k: steps.methods(k) for k in sorted(steps.METHODS)}, width)
    return 0


def cmd_show(args):
    """Print a definition: its file and its JSON (any resource, by path: fec/block/rs/204_188)"""
    from rpps import _config
    domain, _, rest = args.ref.strip("/").partition("/")
    if domain not in _config.domains() or not rest:
        print(f"{args.ref}: give <resource>/<name>, resource one of {_config.domains()}", file=sys.stderr)
        return 1
    file = _config.find(domain, rest)
    if file is None:
        print(f"{args.ref}: not found", file=sys.stderr)
        return 1
    v = _config._variant(file)
    where = f"a variant in {v[0]}" if v is not None else str(file)
    print(f"# {where}" + ("  (extends applied)" if "extends" in _config.read_raw(file) else ""))
    print(_config.format_json(_config.expand(file)))
    return 0


def cmd_rx(args):
    """Receive a capture (or a radio, --device) with a profile: the data bits out, as bytes (most significant bit first)
    to --out; --messages: every frame decoded, one JSON line each (rp.profile.Message); --pcap: every frame as a packet of
    its link type (the profile's capture rules), pcapng to a file, a FIFO, "-" (stdout: `| wireshark -k -i -`) or
    pipe:NAME (a Windows named pipe); --tun / --tap: onto a Linux interface"""
    import numpy as np
    import rpps as rp
    params = _params(args.param)
    sync = {}
    for kv in args.sync:
        kind, method = kv.split("=", 1)
        sync[kind] = {"method": method}
    if (args.file is None) == (args.device is None):
        raise SystemExit("rx: a capture file or --device, one of them")
    if args.messages and args.pcap == "-":
        raise SystemExit("rx: --messages and --pcap - both write to stdout")
    rp.set_effort(args.effort)
    p = rp.profile.load(args.profile, sync=sync or None, **params)
    rx = p.rx(kinds=args.kinds.split(",") if args.kinds else None)
    sps = args.sps or p.params.get("sps", 0.0)
    if args.device:
        src = rp.source.Device(args.device, cf=args.cf, Fs=args.fs, rx_gain=args.gain, sps=sps)
    else:
        src = rp.source.File(args.fmt, args.file, Fs=args.fs, cf=args.cf, sps=sps)
    sinks = []
    if args.pcap:
        sinks.append(rp.source.PcapSink(args.pcap, rpps=args.pcap_rpps, clock=args.pcap_clock))
    for spec, mode in [(t, "tun") for t in args.tun] + [(t, "tap") for t in args.tap]:
        name, _, lt = spec.partition(":")
        sinks.append(rp.source.Tun(name, mode=mode, linktype=lt or None))
    report = sys.stderr if args.pcap == "-" else sys.stdout
    bits = []
    n_msgs = 0
    try:
        for block in src(args.block):
            out = rx(block)
            if args.messages or sinks:
                msgs = rx.take_messages()
                n_msgs += len(msgs)
                if args.messages:
                    for m in msgs:
                        print(json.dumps(m.to_dict()))
                if sinks:
                    packets = p.capture.packets(msgs)
                    for s in sinks:
                        packets >> s
            if isinstance(out, dict): # A profile that branches: one branch's output
                name = args.branch or list(out)[-1]
                if name not in out:
                    raise SystemExit(f"{args.profile} has no branch {name!r} (has {', '.join(out)})")
                out = out[name]
            for o in out if isinstance(out, list) else [out]: # (a profile with a burst step: one output per burst)
                x = np.asarray(o.data)
                bits.append(x < 0 if x.dtype.kind == "f" else x.astype(bool))
    except BrokenPipeError: # The reader (Wireshark) went away
        print("rx: the packet reader closed its end", file=sys.stderr)
    finally:
        for s in sinks:
            s.close()
    bits = np.concatenate(bits) if bits else np.zeros(0, bool)
    what = args.file or args.device
    if args.messages:
        print(f"{what}: {n_msgs} messages, {len(bits)} bits out of {args.profile}", file=sys.stderr)
    else:
        print(f"{what}: {len(bits)} bits out of {args.profile}" + (f" ({args.branch})" if args.branch else "")
              + (f", {n_msgs} messages" if sinks else ""), file=report)
    for s in sinks:
        if isinstance(s, rp.source.Tun) and s.skipped:
            print(f"{s.name}: packets of other link types not written: {s.skipped}", file=report)
    if args.out:
        Path(args.out).write_bytes(np.packbits(bits).tobytes())
        print(f"wrote {args.out}", file=report)
    return 0


def _params(pairs):
    """NAME=VALUE pairs: values as JSON where they parse (numbers, lists, true / false), else as names"""
    out = {}
    for kv in pairs:
        k, v = kv.split("=", 1)
        try:
            out[k] = json.loads(v)
        except ValueError:
            out[k] = v
    return out

def cmd_tx(args):
    """Transmit with a profile: data (a file's bytes, or random) or packets (from a Linux TUN/TAP interface or a pcap, each
    undone by the profile's capture rule) through its transforms to samples, in a file or out of a radio (--device)

    --packet N splits the data into packets of N bytes, each sent as a burst of its own (its pulses complete), --gap
    seconds apart; otherwise it is one stream. A profile with a burst step places the bursts itself (rpps.sync.BurstTx):
    --gap overrides its gap_s, --hop its carrier offsets, --rate-out its output rate (MF-TDMA, hopping). --field
    NAME=VALUE sets the fields of every frame's header (the profile's schedule or defaults otherwise)."""
    import numpy as np
    import rpps as rp
    if (args.out is None) == (args.device is None):
        raise SystemExit("tx: a sample file or --device, one of them")
    p = rp.profile.load(args.profile, **_params(args.param))
    packets = args.from_tun or args.from_tap or args.from_pcap
    from_lt = (args.from_tun or "").partition(":")[2] or ("raw" if args.from_tun else "ethernet" if args.from_tap else None)
    enc = rp.profile.Encoder(p, args.rate, kind=args.kind, fields=_params(args.field) or None, gap=args.gap,
                             hop=[float(f) for f in args.hop.split(",")] if args.hop else None, rate_out=args.rate_out,
                             linktype=from_lt)
    if not enc.bursts and not (args.packet or packets):
        enc.gap = None # (one stream: no silence after it)
    sink, n_out, n_in = None, 0, 0
    def put(o):
        nonlocal sink, n_out
        if sink is None:
            sink = (rp.source.DeviceSink(args.device, cf=args.cf, Fs=o.meta.Fs, tx_gain=args.gain) if args.device
                    else rp.source.Sink(args.fmt, args.out))
        sink.write(o)
        n_out += len(o)
    try:
        if packets:
            if args.from_pcap: # (each packet of its own link type: one the rule's converts from, or the rule's)
                from rpps.link.capture import by_dlt
                items = ((r.data, by_dlt(r.dlt)) for r in rp.source.Pcap(args.from_pcap))
            else:
                name, _, lt = (args.from_tun or args.from_tap).partition(":")
                tun = rp.source.Tun(name, mode="tun" if args.from_tun else "tap", linktype=lt or None)
                items = ((pkt, None) for pkt in tun.packets(timeout=args.timeout))
            for pkt, lt in items:
                put(enc.packet(pkt, linktype=lt))
                n_in += len(pkt)
            put(enc.flush())
        else:
            if args.input:
                data = np.frombuffer(Path(args.input).read_bytes(), np.uint8)
            else:
                data = np.random.default_rng(args.seed).integers(0, 256, args.random, dtype=np.uint8)
            bits = np.unpackbits(data).astype(bool)
            size = 8*args.packet if args.packet else len(bits)
            starts = list(range(0, len(bits), max(size, 1)))
            for k, a in enumerate(starts):
                put(enc.bits(bits[a:a + size], last=k == len(starts) - 1))
            n_in = len(data)
    finally:
        if sink is not None:
            sink.close()
    print(f"{args.out or args.device}: {n_out} samples at {enc.Fs:g} Hz ({n_in} bytes"
          + (f" in {enc.count} packets" if packets else "") + f" through {args.profile})")
    return 0

def cmd_fmt(args):
    """Rewrite config/**/*.json in the house style; with --check only list the files that would change. A "base" the
    folder says too is taken out (one the folder contradicts is an error: move the file)"""
    import rpps # noqa: F401 (registers every resource's loaders)
    from rpps._config import PATH_CONFIG, format_json, join_comments, domains, file_obj
    root = Path(args.dir) if args.dir else PATH_CONFIG
    changed, bad = [], []
    for f in sorted(root.rglob("*.json")):
        obj = json.loads(f.read_text())
        rel = f.relative_to(root).with_suffix("").parts
        if isinstance(obj, dict) and "base" in obj and len(rel) > 1 and rel[0] in domains():
            try:
                obj = file_obj(rel[0], rel[1:], obj)
            except Exception as e:
                bad.append(f"{f.relative_to(root)}: {e}")
                continue
        text = format_json(obj) + "\n"
        assert join_comments(json.loads(text)) == join_comments(obj), f # (a long comment is cut into lines)
        if text != f.read_text():
            changed.append(f)
            if not args.check:
                f.write_text(text)
    for f in changed:
        print(f.relative_to(root))
    for b in bad:
        print(b, file=sys.stderr)
    return 1 if bad or (changed and args.check) else 0


def cmd_survey(args):
    """Survey a capture and print each signal found; --plot also shows the spectrum (needs matplotlib)"""
    import rpps as rp
    src = rp.source.File(args.fmt, args.file, Fs=args.fs, cf=args.cf)
    sv = rp.analysis.survey(src, samples=args.samples)
    print(f"{src.path}: {sv.Fs/1e6:g} MS/s at {sv.cf/1e6:g} MHz, {len(sv)} signals, {sv.bin_hz:.0f} Hz bins, floor {sv.floor:.1f} dB")
    for i, s in enumerate(sv):
        print(f"  {i:>2}  {s.describe()}")
    if args.plot:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("--plot needs matplotlib (pip install rpps[viz])", file=sys.stderr)
            return 1
        sv.plot()
        plt.show()
    return 0


def cmd_profile(args):
    """Profile every signal (or the chosen ones) of a capture: each step that processes it, what it might be where not"""
    import rpps as rp
    src = rp.source.File(args.fmt, args.file, Fs=args.fs, cf=args.cf)
    sv = rp.analysis.survey(src, samples=args.samples)
    idx = None if not args.signals else [int(x) for x in args.signals.split(",")]
    print(f"{src.path}: {len(sv)} signals")
    for i, s in enumerate(sv):
        print(f"  {i:>2}  {s.describe()}")
    out = open(args.json, "a") if args.json else None
    for p in rp.analysis.profile(src, signals=idx, survey=sv, blocks=args.blocks, blind=not args.no_blind):
        print(f"\n{p}")
        if out:
            out.write(json.dumps({"file": str(args.file), "mid": p.signal.mid, "profile": p.to_dict()}, default=str) + "\n")
    if out:
        out.close()
    return 0


def _read_bits(path, fmt):
    import numpy as np
    if fmt == "npy" or path.endswith(".npy"):
        return np.load(path)
    raw = np.fromfile(path, dtype=np.uint8)
    if fmt == "bytes":
        return np.unpackbits(raw)
    return (raw & 1).astype(np.uint8) # One bit a byte (0/1), as rpps rx --out writes unpacked


def cmd_solve(args):
    """What a demodulated stream carries: blind codes, Reed-Solomon, scramblers and PRBS on bits (rpps.analysis.identify.solve,
    scrambler.additive_search); symbols (a complex .npy) are framed first (rpps.analysis.framing.frames)"""
    import numpy as np
    import rpps as rp
    A = rp.analysis
    x = _read_bits(args.file, args.format)
    if np.iscomplexobj(x):
        fr = A.framing.frames.find(x, args.mod)
        print(f"{len(x)} symbols: " + (str(fr) if fr else "no frame structure (no unique word repeats)"))
        if fr is None:
            c = rp.symbol.load_or_generate(args.mod)
            pts = np.asarray(c.points)/np.sqrt(np.mean(np.abs(c.points)**2))
            lab = np.asarray(c.mapping.arr)[np.argmin(np.abs(x[:, None]/np.sqrt(np.mean(np.abs(x)**2)) - pts[None]), axis=1)]
            bits = ((lab[:, None] >> np.arange(c.bits_per_symbol - 1, -1, -1)) & 1).ravel().astype(np.uint8)
        else:
            fb = A.framing.frames.payload_bits(A.framing.frames.cut(x, fr), fr, args.mod)
            rep, same = A.framing.frames.repeats(fb, agree=0.95)
            print(f"  payload {fb.shape[1]} bits a frame, {len(fb)} frames" + (f"; frames repeat every {rep}" if rep else ""))
            if len(fb) >= fb.shape[1] + 48:
                for h, g in A.code.block.linear.solve_block(fb.ravel(), n=fb.shape[1], offset=0):
                    print(f"  frame-aligned block code: {h.name}  (z {h.z:.1f})")
            bits = fb.ravel()
    else:
        bits = np.asarray(x, dtype=np.uint8).ravel()
    print(f"{len(bits)} bits, {bits.mean()*100:.1f}% ones")
    found = A.identify.solve(bits)
    if not found:
        print("  nothing found: no code, RS, scrambler or PRBS (white under every test: encrypted, compressed, or too short)")
    for h, g in found:
        print(f"  {h.kind}: {h.name}  (z {h.z:.1f})" + (f"\n      save to decode with it: {g!r}" if g is not None else ""))
    return 0


_LDPC_RATES = {"B1": "1_4", "B2": "1_3", "B3": "2_5", "B4": "1_2", "B5": "3_5", "B6": "2_3", "B7": "3_4", "B8": "4_5",
               "B9": "5_6", "B10": "8_9", "B11": "9_10"}


def _ldpc_parse(text):
    out = {}
    for m in re.finditer(r"struct DVB_S2_TABLE_(\w+)\s*\{(.*?)\n\};", text, re.S):
        body = m.group(2)

        def arr(name):
            a = re.search(name + r"\[\]\s*=\s*\{(.*?)\};", body, re.S)
            return [int(x) for x in re.findall(r"\d+", a.group(1))] if a else None

        def const(n):
            return int(re.search(rf"static const int {n} = (\d+);", body).group(1))

        out[m.group(1)] = dict(M=const("M"), N=const("N"), K=const("K"), DEG=arr("DEG"), LEN=arr("LEN"), POS=arr("POS"))
    return out


def _ldpc_rows(t):
    pos, out = iter(t["POS"]), []
    for deg, n in zip(t["DEG"], t["LEN"]):
        if deg == 0:
            break
        for _ in range(n):
            out.append([next(pos) for _ in range(deg)])
    return out


def cmd_gen_ldpc(args):
    """Turn DVB-S2 LDPC parity address tables (EN 302 307 annex B/C) into config/fec/block/ldpc files

    The input is xdsopl/LDPC's dvb_s2_tables.hh (https://github.com/xdsopl/LDPC). It is read as data, never executed.
    """
    from rpps._config import PATH_CONFIG
    tables = _ldpc_parse(Path(args.tables).read_text())
    root = Path(args.out) if args.out else PATH_CONFIG / "fec" / "block" / "ldpc"
    root.mkdir(parents=True, exist_ok=True)
    for i in args.ids or ["B5", "B7"]:
        t = tables[i]
        r = _ldpc_rows(t)
        assert len(r) * t["M"] == t["K"], (i, len(r), t["K"])
        name = f"dvbs2_64800_{_LDPC_RATES[i]}"
        (root / f"{name}.json").write_text(json.dumps({
            "base": "block", "type": "ldpc", "n": t["N"], "k": t["K"], "group": t["M"], "systematic": "head", "iters": 50, "alpha": 0.8,
            "comment": f"DVB-S2 normal frame LDPC rate {_LDPC_RATES[i].replace('_', '/')}: parity address table B.{i[1:]} of EN 302 307",
            "table": r}, separators=(",", ":")))
        print(name, "rows", len(r), "k", t["K"])
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="rpps", description="RF Parameter Processor Suite")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_ls = sub.add_parser("ls", help="list the definitions there are, by resource")
    p_ls.add_argument("category", nargs="*", choices=list(CATEGORIES), help="limit output to these categories (default: all)")
    p_ls.add_argument("--tag", help="only the definitions with this tag (a standard or system: dvb-s2, ccsds, ...), across every resource")
    p_ls.add_argument("--tags", action="store_true", help="list the tags and how many definitions carry each")
    p_ls.add_argument("--all", action="store_true", help="also the abstract parents other definitions extend")
    p_ls.add_argument("--blind", action="store_true",
                      help="only what the blind tools (survey, identify, solve) try: no examples, only the selected tags ($RPPS_CONFIG_TAGS)")
    p_ls.set_defaults(fn=cmd_ls)

    p_steps = sub.add_parser("steps", help="list the step kinds and sync methods a profile can use")
    p_steps.set_defaults(fn=cmd_steps)

    p_show = sub.add_parser("show", help="print a definition, e.g. fec/block/rs/204_188")
    p_show.add_argument("ref", help="<resource>/<name>")
    p_show.set_defaults(fn=cmd_show)

    p_rx = sub.add_parser("rx", help="receive a capture file with a profile")
    p_rx.add_argument("profile", help="profile, e.g. dvbs2/normal")
    p_rx.add_argument("file", nargs="?", help="capture file (or --device)")
    p_rx.add_argument("--device", help="receive from a radio instead: a pysdrlib driver or target (hackrf, hackrf@SERIAL)")
    p_rx.add_argument("--gain", type=float, default=None, help="--device: receive gain, a fraction 0-1 of its range")
    p_rx.add_argument("--fs", type=float, required=True, help="sample rate [Hz]")
    p_rx.add_argument("--fmt", default="cf32", help="sample format (default: cf32)")
    p_rx.add_argument("--cf", type=float, default=0.0, help="centre frequency [Hz]")
    p_rx.add_argument("--sps", type=float, default=0.0, help="samples per symbol (default: the profile's sps parameter)")
    p_rx.add_argument("--block", type=int, default=1 << 16, help="samples per block (default: 65536)")
    p_rx.add_argument("--param", action="append", default=[], metavar="NAME=VALUE", help="a profile parameter (repeatable)")
    p_rx.add_argument("--sync", action="append", default=[], metavar="KIND=METHOD", help="a sync method, e.g. timing=known (repeatable)")
    p_rx.add_argument("--effort", default="balanced", choices=("fast", "balanced", "max"),
                      help="speed against recovered SNR (default: balanced; see rpps._effort)")
    p_rx.add_argument("--out", help="write the bits out as bytes here")
    p_rx.add_argument("--branch", help="a profile that branches: the branch whose bits go out (default: the last)")
    p_rx.add_argument("--messages", action="store_true", help="print every frame decoded, one JSON line each (to stdout)")
    p_rx.add_argument("--kinds", metavar="KIND,...", help="only these message kinds, and only the steps they need "
                      "(default: all; rp.profile.info(name).kinds lists them)")
    p_rx.add_argument("--pcap", metavar="TARGET", help="every frame as a packet (the profile's capture rules), pcapng to a "
                      "file, a FIFO, - (stdout: | wireshark -k -i -) or pipe:NAME (Windows: wireshark -k -i \\\\.\\pipe\\NAME)")
    p_rx.add_argument("--pcap-rpps", action="store_true", help="--pcap: every packet in RPPS's header for tools/wireshark/rpps.lua")
    p_rx.add_argument("--pcap-clock", default="message", choices=("message", "wall"),
                      help="--pcap: stamp packets with their message's time (default) or the time they are written")
    p_rx.add_argument("--tun", action="append", default=[], metavar="NAME[:LINKTYPE]",
                      help="write packets of this link type (default raw IP) onto a Linux TUN interface (repeatable)")
    p_rx.add_argument("--tap", action="append", default=[], metavar="NAME", help="write Ethernet packets onto a Linux TAP interface")
    p_rx.set_defaults(fn=cmd_rx)

    p_tx = sub.add_parser("tx", help="transmit data or packets with a profile into a sample file or out of a radio")
    p_tx.add_argument("profile", help="profile, e.g. ble/adv")
    p_tx.add_argument("out", nargs="?", help="sample file to write (or --device)")
    p_tx.add_argument("--device", help="transmit with a radio instead: a pysdrlib driver or target")
    p_tx.add_argument("--cf", type=float, default=None, help="--device: transmit frequency [Hz]")
    p_tx.add_argument("--gain", type=float, default=None, help="--device: transmit gain, a fraction 0-1 of its range")
    p_tx.add_argument("--rate", type=float, required=True, help="rate of the data at the profile's first step (bits or symbols a second)")
    p_tx.add_argument("--fmt", default="cf32", help="sample format (default: cf32)")
    src = p_tx.add_mutually_exclusive_group(required=True)
    src.add_argument("--in", dest="input", help="the data: this file's bytes")
    src.add_argument("--random", type=int, help="the data: this many random bytes")
    src.add_argument("--from-tun", metavar="NAME[:LINKTYPE]", help="packets the system sends on a Linux TUN interface")
    src.add_argument("--from-tap", metavar="NAME", help="Ethernet frames the system sends on a Linux TAP interface")
    src.add_argument("--from-pcap", metavar="FILE", help="the packets of a pcap or pcapng capture (a FIFO is read as written)")
    p_tx.add_argument("--kind", help="packets: the message kind whose capture rule they follow (default: the profile's one)")
    p_tx.add_argument("--timeout", type=float, default=None, help="--from-tun/tap: stop after this many seconds with no packet")
    p_tx.add_argument("--seed", type=int, default=0, help="seed of --random")
    p_tx.add_argument("--packet", type=int, default=0, help="bytes a packet, each sent as a burst of its own")
    p_tx.add_argument("--gap", type=float, default=None, help="seconds between packets (a burst profile: its gap_s otherwise)")
    p_tx.add_argument("--hop", default="", metavar="F1,F2,...", help="a burst profile: carrier offsets [Hz] cycled burst by burst")
    p_tx.add_argument("--rate-out", type=float, default=None, help="a burst profile: output sample rate (the bursts' own otherwise)")
    p_tx.add_argument("--param", action="append", default=[], metavar="NAME=VALUE", help="a profile parameter (repeatable)")
    p_tx.add_argument("--field", action="append", default=[], metavar="NAME=VALUE", help="a header field of every frame (repeatable)")
    p_tx.set_defaults(fn=cmd_tx)

    p_sv = sub.add_parser("survey", help="find and list the signals in a capture file")
    p_sv.add_argument("file", help="capture file")
    p_sv.add_argument("--fs", type=float, required=True, help="sample rate [Hz]")
    p_sv.add_argument("--fmt", default="cf32", help="sample format, see rp.source.formats.ls() (default: cf32, complex float32 pairs)")
    p_sv.add_argument("--cf", type=float, default=0.0, help="centre frequency [Hz] (default: 0)")
    p_sv.add_argument("--samples", type=int, default=2**17, help="samples to read (default: 131072)")
    p_sv.add_argument("--plot", action="store_true", help="show the survey plot (needs matplotlib)")
    p_sv.set_defaults(fn=cmd_survey)

    p_pr = sub.add_parser("profile", help="every step that processes each signal of a capture, and what is unknown")
    p_pr.add_argument("file", help="capture file")
    p_pr.add_argument("--fs", type=float, required=True, help="sample rate [Hz]")
    p_pr.add_argument("--fmt", default="cf32", help="sample format (default: cf32); a gqrx .cs16 is ci16t")
    p_pr.add_argument("--cf", type=float, default=0.0, help="centre frequency [Hz] (default: 0)")
    p_pr.add_argument("--samples", type=int, default=2**17, help="samples the survey reads (default: 131072)")
    p_pr.add_argument("--signals", help="only these, by survey index: 0,3")
    p_pr.add_argument("--blocks", type=int, default=450, help="blocks of 16384 samples each chain runs over (default: 450)")
    p_pr.add_argument("--no-blind", action="store_true", help="skip the blind tools where no definition fits (faster)")
    p_pr.add_argument("--json", help="also append each profile as a JSON line here")
    p_pr.set_defaults(fn=cmd_profile)

    p_so = sub.add_parser("solve", help="what a demodulated bit or symbol stream carries: codes, RS, scramblers, frames")
    p_so.add_argument("file", help="bits (one a byte, or packed with --format bytes), or a .npy of bits or complex symbols")
    p_so.add_argument("--format", choices=["bits", "bytes", "npy"], default="bits", help="how the file holds the bits")
    p_so.add_argument("--mod", default="qpsk", help="constellation of a symbol file (default: qpsk)")
    p_so.set_defaults(fn=cmd_solve)

    p_fmt = sub.add_parser("fmt", help="rewrite config json files in one style")
    p_fmt.add_argument("dir", nargs="?", help="config directory (default: the shipped config)")
    p_fmt.add_argument("--check", action="store_true", help="only list files that would change, exit 1 if any")
    p_fmt.set_defaults(fn=cmd_fmt)

    p_gen = sub.add_parser("gen-ldpc", help="generate DVB-S2 LDPC config files from dvb_s2_tables.hh")
    p_gen.add_argument("tables", help="path to xdsopl/LDPC dvb_s2_tables.hh")
    p_gen.add_argument("ids", nargs="*", help="table ids, e.g. B5 B7 (default: B5 B7)")
    p_gen.add_argument("--out", help="output directory (default: the shipped config/fec/block/ldpc)")
    p_gen.set_defaults(fn=cmd_gen_ldpc)

    args = parser.parse_args(argv)
    return args.fn(args) or 0


if __name__ == "__main__":
    sys.exit(main())
